
import sys
import os
import pandas as pd
import numpy as np
import torch
import numba as nb
from tqdm import tqdm
import networkx as nx
from scipy.spatial.distance import squareform
from networkx.algorithms.clique import find_cliques
import heapq
class Clonotyping:
    def __init__(self,dff,method='heavy',simialrity_criteria=0.8):
        self.df=dff.copy()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.len_seq=0
        self.clustering_criteria=simialrity_criteria
        if 'heavy' in method:
            self.is_heavy=True
            self.df['CDR3'] = self.df['HCDR3']
        else:
            self.is_heavy=False
            self.df['CDR3'] = self.df['LCDR3']

        self.seqs = self.df.drop_duplicates([f"CDR3"])[f"CDR3"].to_list()
        self.len_seq=len(self.seqs[0])
    
    @staticmethod
    def clc_hamming_dist_local(cs_chunk, cs_t, l, m, st, e,device_):
        d = torch.zeros((m, n), dtype=torch.long, device=device_)
        for ll in range(l):
            d += cs_chunk[ll][:, None] != cs_t[ll][None, :]
        return d

    def clc_hamming_dist(self, st):
        e = min(st+chunk_size, n)
        m = e - st
        cs_s_chunk = cs_s[st:st+chunk_size].T.contiguous()
        l_s = self.len_seq
        d = self.clc_hamming_dist_local(cs_s_chunk, cs_s_t, l_s, m, st, e,self.device)
        return d.cpu().numpy()

    def cluster_data(self):
        distance = np.empty((0, n))
        for st in tqdm(range(0, n, chunk_size)):
            e = min(st + chunk_size, n)
            m = e - st
            d = self.clc_hamming_dist(st)
            distance=np.concatenate((distance, d), axis=0)
        epsilon = int(self.len_seq - self.clustering_criteria * self.len_seq)
        distance_matrix = squareform(squareform(distance))
        similar_pairs = np.column_stack(np.where((distance_matrix <= epsilon)))
        ordered_pairs = np.sort(similar_pairs, axis=1)
        # Remove duplicates
        unqiue_pairs = np.unique(ordered_pairs, axis=0)
       
        # Build a graph
        graph = nx.Graph()
        graph.add_edges_from(unqiue_pairs)
        # Extract all maximal cliques (sets of nodes that are fully connected)
        cliques = list(find_cliques(graph))

        # Optionally, filter cliques by size
        min_size = 10  # Example: Only keep cliques with at least 3 members
        refined_clusters = [clique for clique in cliques if len(clique) >= min_size]

        # Assign cluster labels
        # Initialize cluster_labels and use a max heap for sorted clusters
        cluster_labels = {}
        max_heap = []
        assigned_members = set()
        # Build the heap with the clusters and their size as the priority
        for cluster in refined_clusters:
            heapq.heappush(max_heap, (-len(cluster), cluster))  # Use negative length for max-heap behavior

        # Assign cluster labels and dynamically update clusters
        cluster_label_counter = 0
        cluster_size=10
        while max_heap and cluster_size >=10:
            # Pop the largest cluster (based on size)
            cluster_size, cluster = heapq.heappop(max_heap)
            cluster_size = -cluster_size  # Convert back to positive size

            if not cluster:
                continue  # If the cluster is empty after removing assigned members, skip it

            # Assign the remaining members in this cluster
            for member in cluster:
                if member not in assigned_members:
                    cluster_labels[member] = cluster_label_counter
                    assigned_members.add(member)
            cluster_label_counter += 1

            # Now remove the members of the assigned cluster from all remaining clusters
            remaining_clusters = []
            for _, remaining_cluster in max_heap:
                remaining_clusters.append([m for m in remaining_cluster if m not in assigned_members])

            # Rebuild the heap with the updated clusters
            max_heap = []
            for remaining_cluster in remaining_clusters:
                if remaining_cluster:  # Only add non-empty clusters
                    heapq.heappush(max_heap, (-len(remaining_cluster), remaining_cluster))

        seq_indices = list(cluster_labels.keys())
        seq_values = list(cluster_labels.values())
        seq_strings = [self.seqs[i] for i in seq_indices]

        
        # Create DataFrame with results
        column_label=f'clonotype_{self.clustering_criteria}%_cluster_id'
        clustered_df = pd.DataFrame({
            'CDR3': seq_strings,
            column_label: seq_values
        })
        print('cluster_labels',cluster_labels)
        self.df = self.df.merge(clustered_df, on='CDR3', how='left')


    def run(self):
        global n, chunk_size, cs_s, cs_s_t
        n = len(self.seqs)
        chunk_size = min(n, int(1e9 / n))
        cs_s = np.stack([np.asarray(list(s)) for s in self.seqs], axis=0)
        cs_s = (cs_s.view(np.int32) - 65).astype(np.int8)
        cs_s = torch.from_numpy(cs_s)
        cs_s = cs_s.to(self.device)
        cs_s_t = (cs_s.T).contiguous()
        self.cluster_data()
        return self.df