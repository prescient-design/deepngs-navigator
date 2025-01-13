import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,strategies
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import numpy as np
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
AAS = "ACDEFGHIKLMNPQRSTVWY-*"
aa2i = {aa: i for i, aa in enumerate(AAS)}
BOS_ID = 0
EOS_ID = len(AAS) + 1
PAD_ID = len(AAS) + 2
MASK_ID = len(AAS) + 3
aa2i["*"] = MASK_ID
VOCAB_SIZE = len(AAS) + 4
D_EMBED_PER_HEAD = 64
WD = 0.0


class BERT(LightningModule):
    def __init__(self, d_embed=None, nlayer=None, lr=None, output_hidden_states=False):
        super().__init__()
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
            output_hidden_states=output_hidden_states # added after training
            )

        self.lr = lr

        # Initializing a model from the configuration
        self.model = BertForMaskedLM(configuration)

    def forward(self, x):
        out = self.model(**x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=WD)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        out = self(train_batch)
        self.log('train_loss', out.loss)
        return out

    def validation_step(self, val_batch, batch_idx):
        out = self(val_batch)
        self.log('val_loss', out.loss)
        return out

    def test_step(self, test_batch, batch_idx):
        out = self(test_batch)
        self.log('test_loss', out.loss)
        return out


class BertSeqDataset(Dataset):
    def __init__(self, seqs, fixed_masking=False, nmask_min=1, nmask_max=4):
        self.seqs = seqs
        self.fixed_masking = fixed_masking
        self.nmask_max = nmask_max
        self.nmask_min = nmask_min
        if fixed_masking:
            nmasks = np.random.randint(nmask_min, nmask_max, size=len(seqs))
            self.fixed_masks = [np.random.choice(range(len(seq)), size=nmask, replace=False) for nmask, seq in zip(nmasks, seqs)]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        input_ids = torch.LongTensor([BOS_ID] + [aa2i[aa] for aa in seq] + [EOS_ID])
        labels = torch.full_like(input_ids, -100)

        if self.fixed_masking:
            mask_pos = self.fixed_masks[idx] + 1
        else:
            nmask = np.random.randint(self.nmask_min, self.nmask_max)
            mask_pos = np.random.choice(range(len(seq)), size=nmask, replace=False) + 1

        labels[mask_pos] = input_ids[mask_pos]
        input_ids[mask_pos] = MASK_ID

        return input_ids, labels


def batch_converter(seqs):
    batch = []
    for seq in seqs:
        input_ids = torch.LongTensor([BOS_ID] + [aa2i[aa] for aa in seq] + [EOS_ID])
        labels = torch.full_like(input_ids, -100)
        batch.append((input_ids, labels))
    return my_collate(batch)


def my_collate(batch):
    """could be optimized"""
    input_ids, labels = zip(*batch)
    lengths = [len(x) for x in input_ids]
    max_len = max(lengths)
    input_ids      = torch.vstack([torch.cat([x, torch.LongTensor([PAD_ID] * (max_len - l))], dim=0)
                                   for x, l in zip(input_ids, lengths)])
    labels         = torch.vstack([torch.cat([x, torch.LongTensor([-100] * (max_len - l))], dim=0)
                                   for x, l in zip(labels, lengths)])
    attention_mask = torch.vstack([torch.cat([torch.ones(l, dtype=torch.long),
                                              torch.zeros((max_len - l), dtype=torch.long)], dim=0)
                                   for l in lengths])

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels}


class SeqDataModule(LightningDataModule):
    def __init__(self, batch_size=512,seq=[]):
        super().__init__()
        # ---- load seq data
        outdir = "outputs/bert"
        seqs = seq
        print("data_len: ",len(seqs))
        np.random.shuffle(seqs)

        # seqs
        ntest = int(len(seqs)/100)
        nval = int(len(seqs)/100)
        self.seqs_test = BertSeqDataset(seqs[:ntest], fixed_masking=True)
        self.seqs_val = BertSeqDataset(seqs[ntest:ntest+nval], fixed_masking=True)
        self.seqs_train = BertSeqDataset(seqs[ntest+nval:])
        self.batch_size = batch_size
        print('len_seqs',len(seqs))

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        pass

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        pass

    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.seqs_train, batch_size=self.batch_size,
                          collate_fn=my_collate, shuffle=True)

    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.seqs_val, batch_size=self.batch_size,
                          collate_fn=my_collate)

    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.seqs_test, batch_size=self.batch_size,
                          collate_fn=my_collate)



def train_bert_main(
        seqs=[],
        patience=20,
        ENTITY='homa',
        PROJECT='pretraining_deepngs',
        lr=0.0004,
        nlayer=6,
        d_embed=256,
        batch_size=512,
        label='',
        model='',
        num_devices: int = 2
    ):

    NAME = f"bert_{label}_nlayer{nlayer}_lr{lr:.6f}_nembd{d_embed}_bs{batch_size}"
    SAVEDIR = f"pretrained_models/Bert_training_{NAME}"
    os.makedirs(SAVEDIR, exist_ok=True)

    # set up wandb
    wandb_logger = WandbLogger(name=NAME, project=PROJECT, entity=ENTITY)

    # set up model
    if model =='':
        model = BERT(d_embed=d_embed, nlayer=nlayer, lr=lr)

    # setup datamodule
    dm = SeqDataModule(batch_size=batch_size,seq=seqs)

    # setup trainer
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=0.001,
                                        patience=patience,
                                        verbose=True,
                                        mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath=SAVEDIR)
    npz_file_path = f"{SAVEDIR}/best_model_path.npz"

    # Check if the .npz file already exists
    if os.path.exists(npz_file_path):
        try:
            # Load the model path from the .npz file
            loaded_data = np.load(npz_file_path)
            best_path = loaded_data['paths'][0]
            print(f"Loaded model path from {npz_file_path}: {best_path}")
        except:
            best_path=''
    else:
        best_path=''
            
    if best_path == '':
        trainer = Trainer(
                logger=wandb_logger,
                enable_checkpointing=True,
                callbacks=[early_stop_callback, checkpoint_callback],
                default_root_dir=None,
                num_nodes=1,
                enable_progress_bar=True,
                overfit_batches=0.0,
                check_val_every_n_epoch=2,
                max_epochs=400,
                min_epochs=None,
                max_steps=-1,
                max_time="3:00:00:00",  # DD:HH:MM:SS
                limit_train_batches=1.0,
                limit_test_batches=1.0,
                limit_predict_batches=1.0,
                limit_val_batches=1.0,
                val_check_interval=None,
                log_every_n_steps=5,
                accelerator="gpu",
                devices=num_devices,
                strategy=strategies.ddp.DDPStrategy(
                    find_unused_parameters=True
                ) if num_devices > 1 else None,
                sync_batchnorm=False,
                precision=32,
                enable_model_summary=True,
                num_sanity_val_steps=2,
                profiler=None,
                benchmark=False,
                deterministic=False,
                reload_dataloaders_every_n_epochs=0,
            )


        # train
        trainer.fit(model, dm)

        # save best
        best_path = trainer.checkpoint_callback.best_model_path
        np.savez(f"{SAVEDIR}/best_model_path.npz", paths=[best_path])
        print(best_path)
        # test
        trainer.test(model, datamodule=dm)

    # test loading model ckpt
    model = BERT.load_from_checkpoint(best_path, d_embed=d_embed, nlayer=nlayer, lr=lr)

    wandb.finish()
    return model,best_path
