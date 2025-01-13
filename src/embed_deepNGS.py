import os
import pandas as pd
import numpy as np
import torch
import numba as nb
import gc
import time
from tqdm import tqdm
from src.train_bert import BERT, batch_converter,train_bert_main
from src.contrastive_learning import train_model,batch_converter_CL, ContrastiveModel


class DataProcessor:
    def __init__(self, dataframe,desc='test',len_group=''):
        self.df = dataframe
        self.len_group=len_group

    def load_data(self):
        df=self.df
        df['AA'] = df['fv_heavy']+'*'+df['fv_light']
        df['CDR3'] = df['HCDR3']+'*'+df['LCDR3']
        if self.len_group!='':
            df=df[df['V-lengths']==self.len_group]
        df=df[~df['AA'].isna()]
        df=df.drop_duplicates('AA').reset_index(drop=True)
        print('size of loaded df:',len(df))
        return df

class Wandb:
    def __init__(self,wandb_project,wandb_entity):
        # wandb credentials
        self.project = wandb_project
        self.entity = wandb_entity

@nb.jit
def argsort_helper(ir,ic,edits):
    mm = ir.max()
    mn = ir.min()
    ic_condensed = np.zeros_like(ir)
    ee = 0
    for i in range(mn,mm+1):
        initial_=ee
        k = 0
        while ee < len(ir) and ir[ee] == i:
            ic_condensed[ee] = k
            ee += 1
            k += 1
        last_=ee
        arg_sorted=np.argsort(edits[initial_:last_])
        ic[initial_:last_]=np.take(ic[initial_:last_], arg_sorted)
        edits[initial_:last_]=np.take(edits[initial_:last_], arg_sorted)
    return ic_condensed

@nb.jit
def reformat_ir(ir):
    mm = ir.max()
    mn = ir.min()
    range_=np.arange(mn, mm+1, 1)
    start_indices = np.empty_like(range_)
    end_indices = np.empty_like(range_)
    ee = 0
    for i in range_:
        k = 0
        start_indices[i]=ee
        while ee < len(ir) and ir[ee] == i:
            ee += 1
            k += 1
        end_indices[i]=ee
    return start_indices, end_indices

class Main:
    
    def __init__(self,desc,len_group,dataframe):
        self.desc=desc
        self.len_group = len_group
        self.processed_data =  DataProcessor(dataframe,self.desc,self.len_group).load_data()
        self.devices = [torch.device(f'cuda:{i}') for i in range(1)]

    def pretrain_model(self,wandb_project,wandb_entity,bs,lr,patience,d_embed,num_devices):
        df_=self.processed_data.copy()
        _,best_path=train_bert_main(seqs=df_['AA'].values,patience=patience,ENTITY=wandb_entity,PROJECT=wandb_project,
                                    lr=lr,nlayer=6,d_embed=d_embed,batch_size=bs,label=f'{self.desc}',num_devices=num_devices)
        return best_path

    def train_embedding_projecting_model_tSNE(self,pretrained_model_path,n_components=2,perplexity=90):
        from sklearn.manifold import TSNE
        df = self.processed_data
        
        model_pretrained_ = BERT.load_from_checkpoint(pretrained_model_path, d_embed=256, nlayer=6, lr=0.001,output_hidden_states=True).to(self.devices[0])
        for param in model_pretrained_.parameters():
            param.requires_grad = False
        es_all = []
        batch_size=1000
        seqs=df.AA.values
        for st in tqdm(range(0, len(seqs), batch_size)):
            end = min(st + batch_size, len(seqs))
            seqs_tmp = seqs[st:end]
            batch = batch_converter(seqs_tmp)
            with torch.no_grad():
                batch_gpu = {k: v.cuda() for k, v in batch.items()}  # Move data to GPU
                es = model_pretrained_(batch_gpu).hidden_states[-1]
                es = torch.stack([e[1:len(s)+1].sum(dim=0, keepdim=False) for e, s in zip(es, seqs_tmp)]).cpu()
                es_all.append(es)
        es = torch.cat(es_all, dim=0)
        es=es.cpu().numpy()
        u = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=30).fit_transform(es)
        df["e1"] = u[:, 0]
        df["e2"] = u[:, 1]
        return df

    def train_embedding_projecting_model_umap(self,pretrained_model_path,n_neighbors=15,min_dist=1,n_components=2):
        import umap.umap_ as umap
        df = self.processed_data
        model_pretrained_ = BERT.load_from_checkpoint(pretrained_model_path, d_embed=256, nlayer=6, lr=0.001,output_hidden_states=True).to(self.devices[0])
        for param in model_pretrained_.parameters():
            param.requires_grad = False
        es_all = []
        batch_size=1000
        seqs=df.AA.values
        for st in tqdm(range(0, len(seqs), batch_size)):
            end = min(st + batch_size, len(seqs))
            seqs_tmp = seqs[st:end]
            batch = batch_converter(seqs_tmp)
            with torch.no_grad():
                batch_gpu = {k: v.cuda() for k, v in batch.items()}  # Move data to GPU
                es = model_pretrained_(batch_gpu).hidden_states[-1]
                es = torch.stack([e[1:len(s)+1].sum(dim=0, keepdim=False) for e, s in zip(es, seqs_tmp)]).cpu()
                es_all.append(es)
        es = torch.cat(es_all, dim=0)
        es=es.cpu().numpy()
        fit = umap.UMAP(
            n_neighbors=50,
            min_dist=1,
            n_components=2,
            metric="euclidean",
            densmap=False
        )
        u = fit.fit_transform(es)
        df["e1"] = u[:, 0]
        df["e2"] = u[:, 1]
        return df

    def train_embedding_projecting_model_ablang(self):
        import umap.umap_ as umap
        import ablang
        df = self.processed_data
        model_ablang = ablang.pretrained("heavy")
        model_ablang.freeze()
        batch_converter = model_ablang.tokenizer
        model_ablang = model_ablang.AbRep
        model_ablang.to(self.devices[0])
        
        def embed_ablang(model, seqs, batch_size=8):
            """sum embedding"""
            es_all = []
            for st in tqdm(range(0, len(seqs), batch_size)):
                end = min(st + batch_size, len(seqs))
                seqs_tmp = seqs[st:end]
                batch_tokens = batch_converter(seqs_tmp, pad=True)
                with torch.no_grad():
                    results = model(batch_tokens.to(self.devices[0]))
                    es = results.last_hidden_states
                    es = torch.stack([e[1:len(s)+1].sum(dim=0, keepdim=False) for e, s in zip(es, seqs_tmp)]).cpu().numpy()
                    es_all.append(es)
            es = np.concatenate(es_all, axis=0)
            ls = np.asarray([len(s) for s in seqs])
            # compute mean vector
            es_mean = es.sum(axis=0) / np.sum(ls)
            # subtract mean vectors
            es = es - ls.reshape(-1, 1) * es_mean.reshape(1, -1)
            return es
        es = embed_ablang(model_ablang, df.AA.values)
        fit = umap.UMAP(
            n_neighbors=15,
            min_dist=1,
            n_components=2,
            metric="euclidean",
            densmap=False
        )
        u = fit.fit_transform(es)

        df["e1"] = u[:, 0]
        df["e2"] = u[:, 1]
        return df

    def train_embedding_projecting_model(
        self,
        desc: str = "test",
        length: int = -1,
        wandb_project: str = "deepngs",
        wandb_entity: str = "",
        fileName: str = "",
        max_nn: float = 5000.0,
        d_max: int = 7,
        d_max_cdr3: int = 3,
        nn_estimate: int = 10,
        pretrained_model_path: str = "",
        bs: int = 256,
        d_embed: int = 256,
        lr: float = 5e-5,
        model_type: str = "bert",
        patience: float = 20.0,
        min_distance: float = 1.0,
        len_group: str = '',
        max_epochs: int = 300,
        clustering_method: str = "",
        output_name: str = 'test_output', # default test_output, but usually generated based on desc and timestamp
        pretrained_deepngs_model_path: str = "",
        loss_function: str = "",
        num_devices: int = 2,
    ):
         # wandb credentials
        project = wandb_project
        entity = wandb_entity
        # Initialize devices for each GPU
        devices = [torch.device(f'cuda:{i}') for i in range(num_devices)]


        # Load data
        df = self.processed_data
        seqs = df.AA.values
        df['cdr3_'] = df.CDR3.values
        cdrs = df['cdr3_'].values

        # Convert sequences to numpy arrays
        cs = np.stack([np.asarray(list(s)) for s in seqs], axis=0)
        # Prepare the data (on GPU)
        cs = (cs.view(np.int32) - 65).astype(np.int8) # convert to smallest possible integers
        cs = torch.from_numpy(cs)
        cs = cs.to(devices[0]) # send to gpu

        # Transpose once
        cs_t = (cs.T).contiguous()
        n = len(seqs)
        chunk_size = min(n, int(1e8 / n))

        # Initialize list to store number of neighbors
        num_neighbors = []

        # Set maximum distance for close neighbors
        d_max_c = d_max
        d_max_cdr3_c=d_max_cdr3
        n_neighbor_max_close=max_nn

        l = len(seqs[0])

        # Iterate over chunks of sequences to define edit distance cut off if needed and filter out isolated seqeunces
        for st in tqdm(range(0, n, chunk_size)):
            e = min(st+chunk_size, n)
            m = e - st

            # Calculate whole sequence distance
            cs_chunk = cs[st:e].T.contiguous()
            d = torch.zeros((m, n), dtype=torch.long, device=devices[0])

            for ll in range(l):
                d += cs_chunk[ll][:, None] != cs_t[ll][None, :]

            if st==0 and d_max_c==-1:
                # For each sequence, find the nn_estimate nearest neighbors,
                # compute the maximum distance among them, and then take the
                # average of these maximum distances across all sequences.
                sample_size=m
                random_indices = torch.randperm(m)
                max_distances_subset = torch.zeros(sample_size, dtype=torch.float)
                for i, idx in enumerate(random_indices):
                    distances = d[idx]
                    # Find the indices of the `nn_estimate` closest sequences
                    _, indices = torch.topk(distances, k=min(nn_estimate, n), largest=False)

                    # Find the maximum distance among these `nn_estimate` closest sequences
                    max_distances_subset[i] = torch.max(distances[indices])
                # Calculate the average of these maximum distances over the selected subset
                d_max_c = torch.mean(max_distances_subset)+1
                d_max_c=torch.ceil(d_max_c).int()
                d_max_cdr3_c=int(d_max_c/2)
                print ('edit distance cut off defined as:')
                print('full seqeunce: ',d_max_c)
                print('cdr3: ',d_max_cdr3_c)
            # Find close neighbors for whole sequence
            ir_c, _ = torch.nonzero(d <= d_max_c).T
            ir_c = ir_c.cpu().numpy()
            num_neighbors_ = np.bincount(ir_c, minlength=m)
            ir_c += st # add the offset

            # Add to final list of neighbors
            num_neighbors.extend(list(num_neighbors_))

        num_neighbors = np.array(num_neighbors)

        # Clean up memory
        del d
        gc.collect()

        # Filter out isolated sequences 
        df['log10_fullseq_hamming_d'] = np.log10(num_neighbors + 1)
        print('df.shape before filtering isoalyted seqeunces', df.shape)
        df_nn_filtered = df[(df['log10_fullseq_hamming_d'] >= 0.5)] # drop isolated sequences
        print('df.shape after filteration', df_nn_filtered.shape)

        df_nn_filtered=df_nn_filtered.reset_index(drop=True)
        seqs = df_nn_filtered.AA.values
        df_nn_filtered['cdr3_'] = df_nn_filtered.CDR3.values
        cdrs = df_nn_filtered['cdr3_'].values

        # Convert sequences to numpy arrays
        cs = np.stack([np.asarray(list(s)) for s in seqs], axis=0)
        cs_cdr = np.stack([np.asarray(list(s)) for s in cdrs], axis=0)

        # Try to load a pretrained model, if it fails, set the model to None
        if model_type == "bert":
            model_pretrained_ = BERT.load_from_checkpoint(pretrained_model_path, d_embed=256, nlayer=6, lr=lr, output_hidden_states=True).to(self.devices[0])
      
        for param in model_pretrained_.parameters():
            param.requires_grad = False
        ckpt0 = pretrained_model_path
 
        # Prepare the data (on GPU)
        cs = (cs.view(np.int32) - 65).astype(np.int8)  # Convert to smallest possible integers
        cs = torch.from_numpy(cs)
        cs = cs.to(devices[0])  # Send to GPU

        cs_cdr = (cs_cdr.view(np.int32) - 65).astype(np.int8)  # Convert to smallest possible integers
        cs_cdr = torch.from_numpy(cs_cdr)
        cs_cdr = cs_cdr.to(devices[0])  # Send to GPU

        # Transpose once
        cs_t = (cs.T).contiguous()
        cs_cdr_t = (cs_cdr.T).contiguous()
        n = len(seqs)
        chunk_size = min(n, int(1e8 / n))

        # define result lists
        results_c = []
        num_neighbors = []
        l = len(seqs[0])
        l_cdr = len(cdrs[0])

        # Iterate over chunks of seqeunces to find neighbors
        for st in tqdm(range(0, n, chunk_size)):
            # Define the end of the chunk
            e = min(st+chunk_size, n)
            # Get the chunk of data
            cs_chunk_cdr = cs_cdr[st:e].T.contiguous()
            m = e - st

            # Initialize a zero tensor
            d = torch.zeros((m, n), dtype=torch.long, device=devices[0])

            # Compute the distance for each element in the chunk
            for ll in range(l_cdr):
                d += cs_chunk_cdr[ll][:, None] != cs_cdr_t[ll][None, :]

            # Compute close neighbors for cdr
            ir_cdr_c, ic_cdr_c = torch.nonzero(d <= d_max_cdr3_c).T
            edits_cdr_c = d[ir_cdr_c, ic_cdr_c]
            ic_cdr_c = ic_cdr_c.cpu().numpy()
            iselect_cdr_c = (edits_cdr_c <= d_max_cdr3_c).cpu().numpy()

            # Delete unnecessary variables to free memory
            del edits_cdr_c,ir_cdr_c,d
            gc.collect()

            # Compute whole sequence distance
            cs_chunk = cs[st:e].T.contiguous()
            d = torch.zeros((m, n), dtype=torch.long, device=devices[0])

            for ll in range(l):
                d += cs_chunk[ll][:, None] != cs_t[ll][None, :]

            # Compute close neighbors for the whole sequence
            ir_c, ic_c = torch.nonzero(d <= d_max_c).T
            edits_c = d[ir_c, ic_c]
            ir_c = ir_c.cpu().numpy()
            ic_c = ic_c.cpu().numpy()
            edits_c = edits_c.cpu().numpy()
            select_c=np.in1d(ic_c,ic_cdr_c[iselect_cdr_c])
            ir_c=ir_c[select_c]
            ic_c=ic_c[select_c]
            edits_c=edits_c[select_c]
            ic_condensed =argsort_helper(ir_c,ic_c,edits_c)
            num_neighbors_ = np.bincount(ir_c, minlength=m)
            iselect = ic_condensed < n_neighbor_max_close
            ir_c = ir_c[iselect]
            ic_c = ic_c[iselect]
            ic_condensed=ic_condensed[iselect]
            edits_c = edits_c[iselect]
            ir_c += st # add the offset

            # Append the results
            results_c.append((ir_c, ic_c, edits_c))
            num_neighbors.extend(list(num_neighbors_))

        # Concatenate the results
        ir_c, ic_c, edits_c = [np.concatenate(x) for x in zip(*results_c)]
        num_neighbors=np.array(num_neighbors)
        
        # Delete unnecessary variables to free memory
        del iselect,edits_c,ic_cdr_c,ic_condensed,d,results_c
        gc.collect()


        # ---- train contrastive model and embed sequences ----#
        nseq = len(seqs)
        length=len(seqs[0])
        check_val_every_n_epoch=int(max(1, 13. -2. * np.log10(nseq + 1)))
        head_type = 'linear'
        bs = bs
        max_epochs=max_epochs
        max_epochs_step2 =int(max_epochs/2)
        model_type = model_type
        use_fitted_kernel = True
        neighbor_min_dist = min_distance
        name = 'stage1_'+output_name
        s_ir,  e_ir  = reformat_ir(ir_c)
        neighbor_arr  = (s_ir,  e_ir,  ic_c)

        args_dict = dict(desc=desc,model_type=model_type, d_embed=d_embed, nlayer=6, lr=lr,
                        head_type=head_type, neighbor_min_dist=neighbor_min_dist,
                        length=length,
                        use_fitted_kernel=use_fitted_kernel,model_pretrained_=model_pretrained_,loss_function=loss_function)

        store_dir = f"outputs/deepngs/models/{output_name}/"
        os.makedirs(store_dir, exist_ok=True)
        
        ckpt1 = train_model(seqs, neighbor_arr, args_dict, patience=patience,
                    name=name,
                    max_epochs=max_epochs, stage=1, batch_size=bs,
                    store_dir=store_dir,
                    model_ckpt=ckpt0,
                    project=project,
                    entity=entity,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    num_devices=num_devices
                    )
        out_dim = 2
        time.sleep(1.5)
        
        name = 'stage2_'+output_name

        ckpt2 = train_model(seqs, neighbor_arr, args_dict, patience=patience,
                    name=name,
                    max_epochs=max_epochs_step2, stage=2, batch_size=bs,
                    store_dir=store_dir,
                    model_ckpt=ckpt1,
                    project=project,
                    entity=entity,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    num_devices=num_devices
                )
        time.sleep(1.5)
        
        name = 'stage3_'+output_name

        ckpt3 = train_model(seqs, neighbor_arr, args_dict, patience=patience,
                    name=name,
                    max_epochs=max_epochs, stage=3, batch_size=bs,
                    store_dir=store_dir,
                    model_ckpt=ckpt2,
                    project=project,
                    entity=entity,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    num_devices=num_devices
                )
        time.sleep(1.5)

        # ---- load the trained model and calculate embeddings ----#
        model_ckpt = ckpt3
        model = ContrastiveModel.load_from_checkpoint(
                        model_ckpt,
                        **args_dict)
        model.eval()
        model = model.to(devices[0])
        torch.save(model.state_dict(), f'{store_dir}/{name}.ckpt')
        with torch.no_grad():
            ess = []
            for st in range(0, len(seqs), bs):
                e = min(st + bs, len(seqs))
                inputs1, inputs2, mask, mask_repulsion  = batch_converter_CL(seqs[st:e])
                inputs1 = {k: v.to(devices[0]) if type(v) is torch.Tensor else v for k, v in inputs1.items()}
                es = model(inputs1).cpu()
                ess.append(es)
            es = np.concatenate(ess)

        # return embeddings and number of neighbors
        df_emb=pd.DataFrame({'e1':es[:, 0],'e2':es[:, 1],'seq':seqs,'log10_num_neighbors':np.log10(num_neighbors + 1)},columns=['e1','e2','seq','log10_num_neighbors'],index=range(len(seqs)))
        df=pd.merge(df_nn_filtered,df_emb,left_on='AA',right_on='seq',how='left')
        return df