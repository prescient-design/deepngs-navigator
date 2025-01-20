import os
import pandas as pd
import logging
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import leidenalg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import umap.umap_ as umap
from src.utils_deepNGS import ResiConvert,SecondSetGroupConversion
import numpy as np
class LeidenClustering:
    def __init__(self,df,utilise_cdr3=False,sample_size=10000,n_neighbors=5):

        self.utilise_cdr3=utilise_cdr3
        self.sample_size=sample_size
        self.df=df.copy()
        self.n_neighbors_=n_neighbors
        if sample_size==-1:
            if self.df.shape[0]<1000:
                self.n_neighbors_=5
            else:
                self.n_neighbors_=int(np.log10(self.df.shape[0]))
        
    
    def integrate_generalised_cdr3_emebddings(self,df_copy):
        # Step 1: Convert CDR3s into their generalized forms
        cdr3_sequences = df_copy['CDR3'].reset_index().copy()
        cdr3_sequences.columns = ['Title', 'Seq']
        SecondSetGroupConversion(cdr3_sequences)
        df_copy['CDR3_g'] = cdr3_sequences['Seq'].values 
        
        # Step 2: Generate k-mer embeddings for generalized CDR3 sequences
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))  # k-mer sizes (tetragrams)
        kmer_embeddings = vectorizer.fit_transform(df_copy['CDR3_g'].values)
        
        # Step 3: Perform PCA to reduce dimensionality
        pca = PCA(n_components=5)  # Reduce to 5 dimensions
        pca_embeddings = pca.fit_transform(kmer_embeddings.toarray())
        
        # Step 4: Apply UMAP for further dimensionality reduction to 2D
        umap_embeddings = umap.UMAP(
            n_neighbors=self.n_neighbors_, 
            n_components=2, 
            metric='euclidean', 
            random_state=42
        ).fit_transform(pca_embeddings)
        
        # Step 5: Scale UMAP embeddings to match the range of original deepNGS embeddings
        umap_min, umap_max = umap_embeddings.min(), umap_embeddings.max()
        deepngs_min, deepngs_max = df_copy[['e1', 'e2']].min().min(), df_copy[['e1', 'e2']].max().max()
        scaled_umap_embeddings = ((umap_embeddings - umap_min) / (umap_max - umap_min)) * (deepngs_max - deepngs_min) + deepngs_min
        
        # Further scale down the CDR3 UMAP embeddings
        scaled_umap_embeddings *= 0.01  # Optional scaling factor
        
        # Step 6: Integrate the scaled embeddings into the original 'e1', 'e2' coordinates
        df_copy[['e1', 'e2']] += scaled_umap_embeddings
        
        return df_copy


        
    def fit_leiden(self):
        if self.sample_size==-1 or self.sample_size>self.df.shape[0]:
            df_=self.df.copy()
        else:
            df_=self.df.sample( self.sample_size,random_state=42)
            print('sample size:',df_.shape,df_.iloc[0])
        if  self.utilise_cdr3:
            df_=self.integrate_generalised_cdr3_emebddings(df_)
        A = kneighbors_graph(df_[['e1', 'e2']], n_neighbors=self.n_neighbors_, include_self=False).toarray()
        # Convert the adjacency matrix to an igraph Graph
        g = ig.Graph.Adjacency((A > 0).tolist())
        # Perform Leiden clustering
        partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,resolution_parameter=1)

        # Extract cluster labels
        leiden_labels = partition.membership
        
        df_['cluster_id_leiden']=leiden_labels
        self.df=pd.merge(self.df,df_[['AA','cluster_id_leiden']],on='AA',how='left')
        return self.df
        
    