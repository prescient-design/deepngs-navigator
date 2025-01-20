import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BertForMaskedLM, BertConfig
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import numpy as np 
from scipy import  optimize
import wandb
from pytorch_lightning.loggers import WandbLogger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision("medium") # to make lightning happy

AAS = "ACDEFGHIKLMNPQRSTVWY-*"
aa2i = {aa: i for i, aa in enumerate(AAS, 1)}
BOS_ID = 0
EOS_ID = len(AAS) + 1
PAD_ID = len(AAS) + 2
MASK_ID = len(AAS) + 3
aa2i["*"] = MASK_ID
VOCAB_SIZE = len(AAS) + 4
D_EMBED_PER_HEAD = 64

WD = 0.0 # weight decay


class ContrastiveModel(LightningModule):
    def __init__(self, desc, model_type="bert", d_embed=None, 
                 nlayer=None, lr=None, stage=1, 
                 head_type="linear", length=115,
                 neighbor_min_dist=0,
                 use_fitted_kernel=True,model_pretrained_=None,loss_function='diagonal_positive_sum_negative'):

        super().__init__()
        self.desc=desc
        self.loss_function=loss_function
        self.save_hyperparameters()        
        self.d_embed=d_embed
        self.nlayer=nlayer
        self.lr=lr
        self.stage=stage
        self.model_type=model_type
        self.model_pretrained_=model_pretrained_
        self.neighbor_min_dist=neighbor_min_dist
        self.use_fitted_kernel=use_fitted_kernel
        self.L = length # sequence length e.g 128
        self.register_buffer("sig_LM_embedding_dist", torch.ones(self.L, dtype=torch.float32))  # position specific normalization factor
        self.register_buffer("n_LM_embedding_dist", torch.zeros(1, dtype=torch.float32))

        if use_fitted_kernel:
            # source: https://gist.githubusercontent.com/NikolayOskolkov/9d00868443423063f8d9036a31c04f37/raw/94d74db106a53f4a099fdf3c457a911fb2ccb47a/min_dist.py
            def f(x, min_dist):
                y = []
                for i in range(len(x)):
                    if(x[i] <= min_dist):
                        y.append(1)
                    else:
                        y.append(np.exp(- x[i] + min_dist))
                return y
            dist_low_dim = lambda x, a, b: 1 / (1 + a*x**(2*b))
            x = np.linspace(0., 10., 1000)
            p, _ = optimize.curve_fit(dist_low_dim, x, f(x, neighbor_min_dist))
            a, b = p
            self.a=a
            self.b=b

        # Initializing a model from the configuration
        if model_type == "bert":
            nhead = int(d_embed // D_EMBED_PER_HEAD)
            configuration = BertConfig(
                vocab_size=VOCAB_SIZE,
                hidden_size=d_embed,
                num_hidden_layers=nlayer,
                num_attention_heads=nhead,
                intermediate_size=d_embed * 4,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=PAD_ID,
                position_embedding_type="absolute",
                use_cache=True,
                classifier_dropout=None,
                output_hidden_states=True
                )
            self.model = BertForMaskedLM(configuration)
        self.linear_pre = nn.Linear(d_embed, d_embed, bias=True)

        # this allows to share a common base model for different out dim
        self.head_type = head_type
        if head_type == "linear":
            self.head2 = nn.Linear(d_embed, 2, bias=False)
        else:
            raise ValueError
        self.set_stage()

    

    def _freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True

    

    def set_stage(self):
        self._freeze(self.head2)
        if self.stage == 1:
            self._unfreeze(self.model)
            self._unfreeze(self.linear_pre)
        elif self.stage == 2:
            self._unfreeze(self.head2)
            self._freeze(self.model)
            self._freeze(self.linear_pre)
        elif self.stage == 3:
            self._unfreeze(self.head2)
            self._unfreeze(self.model)
            self._unfreeze(self.linear_pre)
        else:
            raise ValueError

        if self.model_type == "bert":
            for param in self.model.cls.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.model_type == "bert":
            out = self.model(**x)
            es = out.hidden_states[-1][:, 0] # grabs BOS token embedding
        else:
            es = self.model(x["input_ids"])

        es = self.linear_pre(es)

        if self.stage in [ 2, 3]:
            es = self.head2(es)
        return es

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=WD)
        return optimizer


    def compute_loss_from_batch(self, batch):
        inputs1, inputs2, mask, mask_neighbors = batch

        es1 = self(inputs1)  # [N, d]
        es2 = self(inputs2)  # [N, d]

        d11 = (es1[:, None] - es1[None, :]).square().sum(-1) + 1e-12
        d12 = (es1[:, None] - es2[None, :]).square().sum(-1) + 1e-12
        d_sq = torch.cat([d11, d12], dim=1)

        if self.use_fitted_kernel:
            d_sq = self.a * d_sq.pow(self.b)


        # distance
        phi = 1. / (1. + d_sq)  # dim [bs, 2 * bs]

        #pos
        bs = mask.size()[0]
        ii = torch.arange(bs)
        mask_pos=mask_neighbors
        if self.loss_function == 'diagonal_positive_sum_negative':
            lpos = -torch.log(phi[ii, ii + bs])
            lneg = torch.log((phi * mask).sum(1))
        elif self.loss_function == 'weighted_by_mask_size_pos_neg':
            lpos = -torch.log((phi*mask_pos).sum(1)/(mask_pos.sum(1)+1e-7)) 
            lneg =torch.log((phi * mask).sum(1)/(mask.sum(1)+1e-7) +0.01)
        loss=(lpos + lneg).mean()

        if torch.isnan(loss).any():
            assert False

        return loss


    def training_step(self, train_batch, batch_idx):
        loss = self.compute_loss_from_batch(train_batch)
        self.log('train_loss', loss, batch_size=len(train_batch[-2]), sync_dist=True)
        return loss
    

    def validation_step(self, val_batch, batch_idx):
        loss = self.compute_loss_from_batch(val_batch)
        self.log('val_loss', loss, batch_size=len(val_batch[-2]), sync_dist=True)
        return

class SeqDataset(Dataset):
    def __init__(self, seqs, neighbors):
        self.seqs = seqs
        self.neighbors = neighbors
        self.tokens = {}

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        i1 = idx
        s1 = self.seqs[i1]

        if len(self.neighbors) == 1:
            ns = self.neighbors[i1]
            i2s = np.where(ns != -1)[0]
            if len(i2s) == 0:
                i2 = i1 # choose self
            else:
                i2 = ns[np.random.choice(i2s, 1)[0]]
        else:
            starts, ends, ic = self.neighbors

            i2s = ns = ic[starts[idx]:ends[idx]]
            if len(i2s) == 0:
                i2 = i1
            else:
                i2 = np.random.choice(i2s, 1)[0]
        if i1 not in self.tokens:
            t1 = torch.LongTensor([BOS_ID] + [aa2i[aa] for aa in s1] + [EOS_ID])
            self.tokens[i1] = t1
        else:
            t1 = self.tokens[i1]

        s2 = self.seqs[i2]

        if i2 not in self.tokens:
            t2 = torch.LongTensor([BOS_ID] + [aa2i[aa] for aa in s2] + [EOS_ID])
            self.tokens[i2] = t2
        else:
            t2 = self.tokens[i2]

        return t1, t2, i1, i2, ns


def my_collate(batch):
    t1, t2, i1, i2, ns = zip(*batch)
    lengths = [len(x) for x in t1]
    max_len = max(lengths)

    # t1, t2 of dim [bs, max_len]
    input_ids1 = torch.vstack([torch.cat([x, torch.LongTensor([PAD_ID] * (max_len - l))], dim=0)
                              for x, l in zip(t1, lengths)])
    attention_mask = torch.vstack([torch.cat([torch.ones(l, dtype=torch.long),
                                              torch.zeros((max_len - l), dtype=torch.long)], dim=0)
                                   for l in lengths])

    inputs1 = {"input_ids": input_ids1,
              "attention_mask": attention_mask,
              "output_hidden_states": True,
              "return_dict": True}

    if i2[0] is not None:
        input_ids2 = torch.vstack([torch.cat([x, torch.LongTensor([PAD_ID] * (max_len - l))], dim=0)
                                  for x, l in zip(t2, lengths)])
        inputs2 = {"input_ids": input_ids2,
                  "attention_mask": attention_mask,
                  "output_hidden_states": True,
                  "return_dict": True}
        # compute neighbor mask (i.e., do not repel neighbors)
        bs = len(ns)
        ii = np.arange(bs)
        # mask = np.ones((bs, 2 * bs), dtype=float)
        ids = np.asarray(i1 + i2)
        mask = np.stack([np.in1d(ids, n, invert=True).astype(float) for n in ns])
        mask[ii, ii] = 0. # remove self
        mask[ii, bs + ii] = 1. # this we need for normalization
        mask = torch.from_numpy(mask)
        mask.requires_grad_(False)
        # -- second mask for positive forces
        mask_neighbors = 1. - mask.clone()
        # neighbors have value 1.
        mask_neighbors[ii, ii] = 0. # exclude self
        mask_neighbors[ii, bs + ii] = 1. # include neighbor
        # exclude neighbor if same as self
        iself = np.asarray([k for k, (e1, e2) in enumerate(zip(i1, i2)) if e1 == e2])
        mask_neighbors[iself, iself] = 0.
        mask_neighbors.requires_grad_(False)

    else:
        mask = None
        mask_neighbors = None
        inputs2 = None

    return inputs1, inputs2, mask, mask_neighbors



def batch_converter_CL(seqs):
    batch = []
    for seq in seqs:
        input_ids = torch.LongTensor([BOS_ID] + [aa2i[aa] for aa in seq] + [EOS_ID])
        batch.append((input_ids, None, None, None, None))
    return my_collate(batch)


class SeqDataModule(LightningDataModule):
    def __init__(self, seqs, neighbors, batch_size):
        super().__init__()
        self.dataset = SeqDataset(seqs, neighbors)
        self.batch_size = batch_size
        starts,ends,_=neighbors
        num_neighbors=np.array(ends) - np.array(starts)
        self.probabilities = [np.log10(n_neighbor) for n_neighbor in num_neighbors]


    def prepare_data(self):
        '''called only once and on 1 GPU'''
        pass

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        pass

    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          collate_fn=my_collate, drop_last=True,sampler=WeightedRandomSampler(self.probabilities, len(self.probabilities)))
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          collate_fn=my_collate, shuffle=True, drop_last=True)


def train_model(seqs, neighbors, args_dict,
                name, stage, store_dir, batch_size=64,
                patience=20,
                max_epochs=100, model_ckpt=None,
                project="deepngs", entity="homa",
                check_val_every_n_epoch=1,
                num_devices: int = 2
                ):
    # set up wandb
    wandb_logger = WandbLogger(name=name, project=project, entity=entity)



    args_dict["stage"] = stage

    # Set up model
    if model_ckpt is not None:
        # Load the pretrained_checkpoint
        pretrained_checkpoint = torch.load(model_ckpt)
        pretrained_state_dict = pretrained_checkpoint["state_dict"]

       
        # Load the adjusted state dictionary into the ContrastiveModel
        model = ContrastiveModel(
             **{**args_dict, "model_pretrained_": None} 
        )

        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state_dict, strict=False)

        # Log mismatched keys
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)

    else:
        model = ContrastiveModel(**args_dict)

    # setup datamodule
    dm = SeqDataModule(seqs, neighbors, batch_size=batch_size)

    # setup trainer
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=0.001,
                                        patience=patience,
                                        verbose=False,
                                        mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=store_dir)


    trainer = Trainer(
                logger=wandb_logger,    # W&B integration
                enable_checkpointing=True,
                callbacks=[early_stop_callback, checkpoint_callback],
                default_root_dir=None,
                num_nodes=1,
                enable_progress_bar=True,
                overfit_batches=0.0,
                check_val_every_n_epoch=2,
                max_epochs=max_epochs,
                min_epochs=None,
                max_steps=-1,
                min_steps=-1,
                max_time="3:00:00:00", # DD:HH:MM:SS
                limit_train_batches=1.0,
                limit_test_batches=1.0,
                limit_predict_batches=1.0,
                limit_val_batches=1.0,
                val_check_interval=None,
                log_every_n_steps=1,
                accelerator='gpu',
                devices=num_devices, 
                strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True) if num_devices > 1 else 'auto',
                sync_batchnorm=False,
                precision='32',
                enable_model_summary=True,
                num_sanity_val_steps=2,
                profiler=None,
                benchmark=False,
                deterministic=False,
                reload_dataloaders_every_n_epochs=0,)

    # train
    trainer.fit(model, dm)

    # save best
    best_path = trainer.checkpoint_callback.best_model_path

    wandb.finish()

    return best_path
