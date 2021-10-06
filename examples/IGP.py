import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
import copy
import sys
import os
import time
import argparse
import json
import numpy as np
import numpy.linalg as la
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from scipy.sparse import csgraph
from torch.backends import cudnn
from torch.optim import lr_scheduler
from utils import *
from graphConvolution import *

#hyperparameters
num_node = 2708
num_class = 7
num_aval = 100
num_coreset = 18*num_class*(num_class-1)
#num_coreset = 20
batch_size = 5

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
#num_coreset = int((num_node-1500)*0.01)
hidden_size = 128
num_val = 500
num_test = 1000

    
def get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score): 
    receptive_vector=((adj_matrix2[selected_node]))+0
    count=weighted_score.dot(receptive_vector)
    return count

def get_current_neighbors_dense(cur_nodes):
    if np.array(cur_nodes).shape[0]==0:
        return 0
    neighbors=(adj_matrix2[list(cur_nodes)].sum(axis=0)!=0)+0
    return neighbors

def get_current_neighbors_1(cur_nodes):
    if np.array(cur_nodes).shape[0]==0:
        return 0
    neighbors=(adj_matrix[list(cur_nodes)].sum(axis=0)!=0)+0
    return neighbors

def get_entropy_contribute(npy_m1,npy_m2):
    entro1 = 0
    entro2 = 0
    for i in range(npy_m1.shape[0]):
        entro1 -= np.sum(npy_m1[i]*np.log2(npy_m1[i]))
        entro2 -= np.sum(npy_m2[i]*np.log2(npy_m2[i]))
    return entro1 - entro2

def get_max_info_entropy_node_set(idx_used,high_score_nodes):
    max_info_node_set = [] 
    high_score_nodes_ = copy.deepcopy(high_score_nodes)
    labels_ = copy.deepcopy(labels)
    for k in range(batch_size):
        score_list = np.zeros(len(high_score_nodes_))      
        for i in range(len(high_score_nodes_)):
            labels_tmp = copy.deepcopy(labels_)          
            node = high_score_nodes_[i]
            node_neighbors = get_current_neighbors_dense([node])
            adj_neigh = adj_matrix2[list(node_neighbors)]
            aay = np.matmul(adj_neigh,labels_)
            total_score = 0
            for j in range(num_class):
                if model_prediction[node][j] != 0:
                    labels_tmp[node] = 0
                    labels_tmp[node][j] = 1
                    aay_ = np.matmul(adj_neigh,labels_tmp)
                    total_score += model_prediction[node][j]*get_entropy_contribute(aay,aay_)
            score_list[i] = total_score
        idx = np.argmax(score_list)
        max_node = high_score_nodes_[idx]
        max_info_node_set.append(max_node)
        labels_[max_node] = model_prediction[max_node]
        high_score_nodes_.remove(max_node)   
    return max_info_node_set

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid,bias=True)
        self.gc2 = GraphConvolution(nhid, nclass,bias=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
   
def train(epoch, model,record,optimizer):

    model.train()
    optimizer.zero_grad()
    output = model(features_GCN, adj)
    loss_train1 = F.cross_entropy(output[idx_train1], tmp_labels[idx_train1])
    loss_train2_f = nn.KLDivLoss()
    loss_train2 = loss_train2_f(F.log_softmax(output[idx_train2],dim=1), F.softmax(labels[idx_train2],dim=1))
    loss_train = loss_train1+loss_train2
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features_GCN, adj)

    loss_val = F.cross_entropy(output[idx_val], tmp_labels[idx_val])
    acc_val = accuracy(output[idx_val], tmp_labels[idx_val])
    loss_test = F.cross_entropy(output[idx_test], tmp_labels[idx_test])
    acc_test = accuracy(output[idx_test], tmp_labels[idx_test])
    record[acc_val.item()] = acc_test.item()

def update_model_prediction():
    model = GCN(nfeat=features_GCN.shape[1],
            nhid=hidden_size,
            nclass=num_class,
            dropout=0.85)
    model.cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.05, weight_decay=5e-4)
    record = {}
    for epoch in range(400):
        train(epoch,model,record,optimizer)
    output = model(features_GCN, adj)
    sfl = nn.Softmax(dim=1)
    output_ = sfl(output)
    model_prediction = np.array(output_.detach().cpu())
    return model_prediction
    


#read dataset
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="cora")
tmp_labels = copy.deepcopy(labels)
tmp_labels = tmp_labels.cuda()
num_zeros = np.zeros(num_node)
num_ones = np.ones(num_node)
idx_val = list(idx_val.cpu())
idx_test = list(idx_test.cpu())
idx_avaliable = list()
for i in range(num_node):
    if i not in idx_val and i not in idx_test:
        idx_avaliable.append(i)

#compute normalized distance
adj = aug_normalized_adjacency(adj)
adj_matrix = torch.FloatTensor(adj.todense()).cuda()
adj_matrix2 = torch.mm(adj_matrix,adj_matrix).cuda()
adj_matrix2 = np.array(adj_matrix2.cpu())
adj = sparse_mx_to_torch_sparse_tensor(adj).float().cuda()
features_GCN = copy.deepcopy(features) 
features_GCN = torch.FloatTensor(features_GCN).cuda()

model_prediction = np.full((num_node,num_class),1/num_class)
labels = np.full((num_node,num_class),1/num_class)

adj_matrix = np.array(adj_matrix.cpu())
degree_result = []
nodes_degree = []
count = 0
for i in range(num_node):
    tmp_list = []
    tmp_list.append(i)
    cur_neighbors = get_current_neighbors_1(tmp_list)
    tmp_degree = float(np.ones(num_node).dot(cur_neighbors))
    nodes_degree.append([i,tmp_degree])
nodes_degree.sort(key = lambda x:x[1])
for i in range(num_node):
    if nodes_degree[num_node-i-1][0] in idx_avaliable:
        degree_result.append(nodes_degree[num_node-i-1][0])
label_flag = np.ones((num_node,num_class))
degree_flag = 0
idx_train = random.sample(degree_result,2*num_class)
for node in idx_train:
    degree_result.remove(node)
    label_ = tmp_labels[node].item()
    labels[node] = 0
    labels[node][label_] = 1
idx_train1 = []
idx_train2 = []
for node in idx_train:
    if np.max(labels[node]) == 1:
        idx_train1.append(node)
    else:
        idx_train2.append(node)
idx_train1 = torch.LongTensor(idx_train).cuda()
idx_train2 = torch.LongTensor(idx_train).cuda()

idx_avaliable = degree_result[0:num_aval]
idx_avaliable_temp = copy.deepcopy(idx_avaliable)
labels = torch.FloatTensor(labels).cuda()
idx_train = torch.LongTensor(idx_train).cuda()
idx_val = torch.LongTensor(idx_val).cuda()
idx_test = torch.LongTensor(idx_test).cuda()
count = 0
break_list = []
model_prediction = update_model_prediction()
idx_train = list(idx_train.cpu())
labels = np.array(labels.cpu())
while True:
    t1 = time.time()
    max_info_entropy_node_set = get_max_info_entropy_node_set(idx_train,idx_avaliable_temp) 
    cnt1 = 0
    for i in range(batch_size):      
        max_info_entropy_node = max_info_entropy_node_set[i]
        if max_info_entropy_node not in idx_train:
            idx_train.append(max_info_entropy_node)        
        if count%(num_class*(num_class-1))==0:
            idx_train1 = []
            idx_train2 = []
            for node in idx_train:
                if np.max(labels[node]) == 1:
                    idx_train1.append(node)
                else:
                    idx_train2.append(node)
            break_list.append([len(idx_train),len(idx_train1),len(idx_train2)])
        count += 1     
        tmp_class = np.argmax(model_prediction[max_info_entropy_node])
        if tmp_class == tmp_labels[max_info_entropy_node].item():
            labels[max_info_entropy_node] = 0
            labels[max_info_entropy_node][tmp_class] = 1
            idx_avaliable.remove(max_info_entropy_node)
            idx_avaliable_temp.remove(max_info_entropy_node)
            cnt1+=1
        else:
            label_flag[max_info_entropy_node][tmp_class] = 0
            pred_vec = model_prediction[max_info_entropy_node]*label_flag[max_info_entropy_node]
            pred_sum = np.sum(pred_vec)
            for j in range(num_class):
                labels[max_info_entropy_node][j] = pred_vec[j]/pred_sum
            if np.max(labels[max_info_entropy_node]) == 1:
                idx_avaliable.remove(max_info_entropy_node)
                idx_avaliable_temp.remove(max_info_entropy_node)
                cnt1+=1
    idx_train1 = []
    idx_train2 = []
    for node in idx_train:
        if np.max(labels[node]) == 1:
            idx_train1.append(node)
        else:
            idx_train2.append(node)
    idx_train1 = torch.LongTensor(idx_train1).cuda()
    idx_train2 = torch.LongTensor(idx_train2).cuda()
    labels = torch.FloatTensor(labels).cuda()
    idx_train = torch.LongTensor(idx_train).cuda()
    model_prediction = update_model_prediction()
    model_prediction = model_prediction * label_flag
    idx_train = list(idx_train.cpu())
    for i in range(cnt1):
        tmp_node = degree_result[degree_flag]
        degree_flag += 1
        idx_avaliable.append(tmp_node)
        idx_avaliable_temp.append(tmp_node)
    labels = np.array(labels.cpu())
    if count >= num_coreset:
        break


idx_train1 = []
idx_train2 = []
for node in idx_train:
    if np.max(labels[node]) == 1:
        idx_train1.append(node)
    else:
        idx_train2.append(node)
idx_train1 = torch.LongTensor(idx_train1).cuda()
idx_train2 = torch.LongTensor(idx_train2).cuda()
labels = torch.FloatTensor(labels).cuda()
idx_train = torch.LongTensor(idx_train).cuda()
print('xxxxxxxxxx Evaluation begin xxxxxxxxxx')
t_total = time.time()
record = {}
model = GCN(nfeat=features_GCN.shape[1],
        nhid=hidden_size,
        nclass=num_class,
        dropout=0.85)
model.cuda()
optimizer = optim.Adam(model.parameters(),
                        lr=0.05, weight_decay=5e-4)
for epoch in range(400):
    train(epoch,model,record,optimizer)

bit_list = sorted(record.keys())
bit_list.reverse()
for key in bit_list[:10]:
    value = record[key]
    print(key,value)
print('xxxxxxxxxx Evaluation end xxxxxxxxxx')