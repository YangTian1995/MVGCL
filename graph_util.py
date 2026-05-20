import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from math import log

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


def common_neighbor(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    return len(n1&n2)

def admic_adar(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    res=0.0
    if len(n1&n2)==0:
        return 0.0
    for k in n1&n2:
        if len(indices[k])>1:
            res+=1/log(len(indices[k]))
    return res

def jaccard(indices,i,j):
    n1=set(indices[i])
    n2=set(indices[j])
    if len(n1)==0 or len(n2)==0:
        return 0.0
    return len(n1&n2)/len(n1|n2)

def edge_feature(node_num, node_node):
    node_node=sp.csr_matrix(node_node)
    degrees=np.ravel(np.sum(node_node,axis=1))
    indptr = node_node.indptr
    indices = node_node.indices
    split_indices = np.split(indices, indptr[1:-1])
    efeature=np.zeros((node_num,node_num,3),dtype=np.float32)
    sumf=np.zeros(shape=(3,),dtype=np.float32)
    for i in range(node_num):
        fea = [ common_neighbor(split_indices, i, i), \
                admic_adar(split_indices, i, i), jaccard(split_indices, i, i)]
        # print(fea)
        efeature[i, i] = fea
        sumf += fea
        for j in split_indices[i]:
                fea=[common_neighbor(split_indices,i,j),\
                 admic_adar(split_indices,i,j),jaccard(split_indices,i,j)]
                # print(fea)
                efeature[i,j]=fea
                sumf+=fea
    sumf=np.reciprocal(sumf)
    efeature=efeature*sumf
    efeature = np.mean(efeature, axis=2)
    return efeature



#--------------------------------------------------------------------
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def link_prediction(data, node_embedding_matrix, mode="train"):
    if mode == "train":
        pos_edge_index = data.train_pos_edge_index
        neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                           # num_nodes=self.graph_data.num_nodes,
                                           num_neg_samples=data.train_pos_edge_index.size(1),
                                           force_undirected=True).to(device)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        # edge_index = data.train_edge_index

    elif mode == "val":
        edge_index = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=-1)

    elif mode == "test":
        edge_index = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1)

    source_node_embedding = node_embedding_matrix[edge_index[0]]
    target_node_embedding = node_embedding_matrix[edge_index[1]]
    link_predict = (source_node_embedding * target_node_embedding).sum(dim=-1)
    link_predict = link_predict.sigmoid()

    return link_predict
