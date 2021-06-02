from helper import to_device, make_dataset_1M, create_optimizer, ltensor, collate_fn, get_logger

import logging
from logging import config
import json
import random
from random import shuffle
import argparse
from pprint import pprint
from pathlib import Path
import sys
import os
import gc
import math
import ipdb
from tqdm import tqdm
import numpy as np
import pandas as pd
import adabound
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GINConv, GATConv
import itertools
import warnings


class GNN(torch.nn.Module):
    def __init__(self, embed, gnn_layers, gnn_type, device):
        super(GNN, self).__init__()
        h = embed

        def get_layer(gnn_type):
            if gnn_type == 'ChebConv': layer = ChebConv(h, h, K=2)
            elif gnn_type == 'GCNConv': layer = GCNConv(h,h)
            elif gnn_type == 'GINConv': 
                dnn = nn.Sequential(nn.Linear(h, h),
                                        nn.LeakyReLU(),
                                        nn.Linear(h, h))
                layer = GINConv(dnn)
            elif gnn_type == 'SAGEConv':
                layer = SAGEConv(h,h, normalize=True)
            elif gnn_type == 'GATConv':
                layer = GATConv(h,h)
            else: raise NotImplementedError
            return layer

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        if gnn_layers >= 1:
            self.conv1 = get_layer(gnn_type)
        if gnn_layers >= 2:
            self.conv2 = get_layer(gnn_type)
        if gnn_layers == 3:
            self.conv3 = get_layer(gnn_type)

    def forward(self, embeddings, edge_index):

        for layer in [self.conv1, self.conv2, self.conv3]:
            if layer is not None:
                embeddings = layer(embeddings, edge_index)

        return embeddings

class GradReverse(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, inputs):
        return GradReverse.apply(inputs)

class SharedBilinearDecoder(nn.Module):
    """
    Decoder where the relationship score is given by a bilinear form
    between the embeddings (i.e., one learned matrix per relationship type).
    """
    def __init__(self, num_relations, num_weights, embed_dim):
        super(SharedBilinearDecoder, self).__init__()
        self.rel_embeds = nn.Embedding(num_weights, embed_dim * embed_dim)
        self.weight_scalars = nn.Parameter(torch.Tensor(num_weights, num_relations))
        stdv = 1. / math.sqrt(self.weight_scalars.size(1))
        self.weight_scalars.data.uniform_(-stdv, stdv)
        self.embed_dim = embed_dim
        self.num_weights = num_weights
        self.num_relations = num_relations
        self.nll = nn.NLLLoss()
        self.mse = nn.MSELoss()

    def predict(self, embeds1, embeds2):
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, \
                                                     self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        outputs = F.log_softmax(logit, dim=1)
        preds = 0
        for j in range(0, self.num_relations):
            index = (torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            preds += (j + 1) * torch.exp(torch.index_select(outputs, 1, index))
        return preds

    def forward(self, embeds1, embeds2, rels):
        basis_outputs = []
        for i in range(0, self.num_weights):
            index = (torch.LongTensor([i])).cuda()
            rel_mat = self.rel_embeds(index).reshape(self.embed_dim, \
                                                     self.embed_dim)
            u_Q = torch.matmul(embeds1, rel_mat)
            u_Q_v = (u_Q * embeds2).sum(dim=1)
            basis_outputs.append(u_Q_v)
        basis_outputs = torch.stack(basis_outputs, dim=1)
        logit = torch.matmul(basis_outputs, self.weight_scalars)
        outputs = F.log_softmax(logit, dim=1)
        log_probs = torch.gather(outputs, 1, rels.unsqueeze(1))
        loss = self.nll(outputs, rels)
        preds = 0
        for j in range(0, self.num_relations):
            index = (torch.LongTensor([j])).cuda()
            ''' j+1 because of zero index '''
            preds += (j + 1) * torch.exp(torch.index_select(outputs, 1, index))
        return loss, preds

class NodeClassifier(nn.Module):
    def __init__(self, embed_dim, embeddings):
        super(NodeClassifier, self).__init__()
        self.embeddings = embeddings
        self.mode = None
        h = embed_dim

        self.age = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 7))

        self.occupation = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 21))

        self.gender = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, features):
        (user, gender, occupation, age) = features

        embeddings = self.embeddings[user]

        age_pred = self.age(embeddings)
        gender_pred = self.gender(embeddings)
        occupation_pred = self.occupation(embeddings)

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        if (self.mode == 'gender'):
            loss = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss = fn_occupation(occupation_pred, occupation)

        return loss, [age_pred, gender_pred, occupation_pred]

class NodeClassifierWithNoise(nn.Module):
    def __init__(self, embed_dim, embeddings):
        super(NodeClassifierWithNoise, self).__init__()
        self.embeddings = embeddings
        self.mode = None
        h = embed_dim

        self.age = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 7))

        self.occupation = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 21))

        self.gender = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def gaussian(self, tensor, mean=0, stddev=0.75):
        noise = mean + stddev* torch.randn(tensor.size()).cuda()
        return tensor + noise

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, features):
        (user, gender, occupation, age) = features

        embeddings = self.embeddings[user]
        embeddings = self.gaussian(embeddings)

        age_pred = self.age(embeddings)
        gender_pred = self.gender(embeddings)
        occupation_pred = self.occupation(embeddings)

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        if (self.mode == 'gender'):
            loss = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss = fn_occupation(occupation_pred, occupation)

        return loss, [age_pred, gender_pred, occupation_pred]

class SimpleGCMC(nn.Module):
    def __init__(self, decoder, embed_dim, num_ent, encoder=None):
        super(SimpleGCMC, self).__init__()
        self.decoder = decoder
        self.num_ent = num_ent
        self.embed_dim = embed_dim
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)
        if encoder is None:
            r = 6 / np.sqrt(self.embed_dim)
            self.encoder = nn.Embedding(self.num_ent, self.embed_dim, \
                                        max_norm=1, norm_type=2)
            self.encoder.weight.data.uniform_(-r, r).renorm_(p=2, dim=1, maxnorm=1)
        else:
            self.encoder = encoder
        self.all_nodes = to_device(torch.LongTensor(list(range(9992))))

    def encode(self, nodes):     
        embs = self.encoder(self.all_nodes)
        embs = self.batchnorm(embs)
        if ( embs is None ): return embs
        return embs[nodes]

    def predict_rel(self, heads, tails_embed):
        with torch.no_grad():
            head_embeds = self.encode(heads)
            preds = self.decoder.predict(head_embeds, tails_embed)
        return preds

    def forward(self, pos_edges, weights=None, return_embeds=False):
        pos_head_embeds = self.encode(pos_edges[:, 0])
        pos_tail_embeds = self.encode(pos_edges[:, -1])
        rels = pos_edges[:, 1]
        loss, preds = self.decoder(pos_head_embeds, pos_tail_embeds, rels)
        if return_embeds:
            return loss, preds, pos_head_embeds, pos_tail_embeds
        else:
            return loss, preds

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class GAL(SimpleGCMC):
    def __init__(self, decoder, embed_dim, num_ent, edges, args, encoder=None):
        super(GAL, self).__init__(decoder, embed_dim, num_ent, encoder=None)
        
        self.gnn = GNN(self.embed_dim, args.gnn_layers, args.gnn_type, args.device)
        self.edges = edges.cuda()

        h = embed_dim

        self.age = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 7))

        self.occupation = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 21))

        self.gender = nn.Sequential(

            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        self.reverse = GradientReversalLayer()

    def encode(self, nodes):
        embs = self.encoder(self.all_nodes)
        embs = self.batchnorm(embs)
        embs = self.gnn(embs, self.edges)

        if (embs is None): return embs
        return embs[nodes]

    def forward_attr(self, user_features, weights=None, return_embeds=False,):
        (users, gender, occupation, age) = user_features
        user_embeds = self.reverse(self.encode(users))

        #fn_gender = nn.BCEWithLogitsLoss()
        #fn_age = nn.CrossEntropyLoss()
        #fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if (self.mode == 'gender'):
            attack_prediction, labels = (gender_pred, gender.view(-1))
        elif (self.mode == 'age'):
            attack_prediction, labels = (age_pred, age)
        elif (self.mode == 'occupation'):
            attack_prediction, labels = (occupation_pred, occupation)

        one_hot = torch.cuda.FloatTensor(attack_prediction.size(0), attack_prediction.size(1)).zero_()
        mask = one_hot.scatter_(1, labels.view(-1,1), 1)
    
        nonzero = mask * attack_prediction
        avg = torch.mean(nonzero, dim = 0)
        #print(avg)
        loss_adv = torch.abs(torch.max(avg) - torch.min(avg))

        return loss_adv, (age_pred, gender_pred, occupation_pred)

    def set_mode(self, mode):
        self.mode = mode

class GNNWithNoise(SimpleGCMC):
    def __init__(self, decoder, embed_dim, num_ent, edges, args, encoder=None):
        super(GNNWithNoise, self).__init__(decoder, embed_dim, num_ent, encoder=None)
        
        self.gnn = GNN(self.embed_dim, args.gnn_layers, args.gnn_type, args.device)
        self.edges = edges.cuda()

        h = embed_dim

        self.age = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 7))

        self.occupation = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 21))

        self.gender = nn.Sequential(

            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )


        self.reverse = GradientReversalLayer()

    def gaussian(self, tensor, mean=0, stddev=0.75):
        noise = mean + stddev* torch.randn(tensor.size()).cuda()
        return tensor + noise

    def encode(self, nodes):
        embs = self.encoder(self.all_nodes)
        embs = self.batchnorm(embs)
        embs = self.gnn(embs, self.edges)

        if (embs is None): return embs
        return self.gaussian(embs[nodes])

    def forward_attr(self, user_features, weights=None, return_embeds=False,):
        (users, gender, occupation, age) = user_features
        user_embeds = self.reverse(self.encode(users))

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if (self.mode == 'gender'):
            loss_adv = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss_adv = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss_adv = fn_occupation(occupation_pred, occupation)

        return loss_adv, (age_pred, gender_pred, occupation_pred)

    def set_mode(self, mode):
        self.mode = mode

class NeighborClassifier(nn.Module):
    def __init__(self, embed_dim, embeddings, edges):
        super(NeighborClassifier, self).__init__()
        self.embeddings = embeddings
        self.mode = None
        h = embed_dim

        self.age = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 7))

        self.occupation = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 21))

        self.gender = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        edges_np = edges.numpy()
        edge_list = []
        for i in tqdm(range(edges_np.shape[1])):
            edge_list.append((edges_np[0,i], edges_np[1,i]))
        self.G = nx.Graph(edge_list)

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, features):
        (user, gender, occupation, age) = features

        neighbor = []

        user_np = list(user.cpu().numpy())

        for user in user_np:
            for i in self.G.neighbors(user):
                neighbor.append(i)
                break

        embeddings = self.embeddings[neighbor, :]

        age_pred = self.age(embeddings)
        gender_pred = self.gender(embeddings)
        occupation_pred = self.occupation(embeddings)

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        if (self.mode == 'gender'):
            loss = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss = fn_occupation(occupation_pred, occupation)

        return loss, [age_pred, gender_pred, occupation_pred]

class GAL_Neighbor(GAL):
    def __init__(self, decoder, embed_dim, num_ent, edges, args, encoder=None):
        super(GAL_Neighbor, self).__init__(decoder, embed_dim, num_ent, edges, args, encoder=None)
        
        edges_np = edges.numpy()
        edge_list = []
        for i in tqdm(range(edges_np.shape[1])):
            edge_list.append((edges_np[0,i], edges_np[1,i]))
        self.G = nx.Graph(edge_list)
   
    def forward_attr(self, user_features, weights=None, return_embeds=False,):
        (users, gender, occupation, age) = user_features

        neighbor = []

        user_np = list(users.cpu().numpy())

        for user in user_np:
            for i in self.G.neighbors(user):
                neighbor.append(i)
                break
        
        neighbor = torch.tensor(np.array(neighbor)).cuda()
                
        user_embeds = self.reverse(self.encode(neighbor))

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if (self.mode == 'gender'):
            loss_adv = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss_adv = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss_adv = fn_occupation(occupation_pred, occupation)

        return loss_adv, (age_pred, gender_pred, occupation_pred)


class GAL_Nhop(GAL):
    def __init__(self, decoder, embed_dim, num_ent, edges, args, encoder=None, hop=2):
        super(GAL_Nhop, self).__init__(decoder, embed_dim, num_ent, edges, args, encoder=None)
        
        edges_np = edges.numpy()
        print(edges_np.max())
        edge_list = []
        for i in tqdm(range(edges_np.shape[1])):
            edge_list.append((edges_np[0,i], edges_np[1,i]))
        self.G = nx.Graph(edge_list)
        self.hop = hop

    def forward_attr(self, user_features, weights=None, return_embeds=False,):
        (users, gender, occupation, age) = user_features

        neighbor_sub = []

        user_np = list(users.cpu().numpy())
        
        k=0
        include_indices=[]
        ign_indices=[]
        ign_str = "failed vertices: "
        for user in user_np:
            final = user
            path = dict()
            for _ in range(self.hop):
                neighbor = list(self.G.neighbors(final))
                L = lambda a: len(a)
                orig_len = L(neighbor)

                cond = True
                cand = neighbor.pop(np.random.randint(0, L (neighbor)))
                if (cand in path):

                    # This is bounded by O(self.hop)
                    while(L(neighbor) > 0):
                        cand = neighbor.pop(np.random.randint(0, L (neighbor)))
                        if (cand not in path):
                            break
                    if (L(neighbor)==0):
                        ign_str+= "[{} <= {}]".format(orig_len, len(path))
                        cond = False
                        break
                            
                if cond:
                    path[cand]=0
                    final = cand
                else:
                    break
            if cond:
                neighbor_sub.append(final)
                include_indices.append(k)
            else:
                ign_indices.append(k)
            k+=1


        include_indices= np.array(include_indices)
        if (len(ign_indices)>0): warnings.warn("ignoring {} from {}".format(ign_indices, ign_str), RuntimeWarning, stacklevel=2)
        gender= gender[include_indices]
        age = age[include_indices]
        occupation=occupation[include_indices]

        neighbor = torch.tensor(np.array(neighbor_sub)).cuda()

        user_embeds = self.reverse(self.encode(neighbor))

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        gender_pred = self.gender(user_embeds)
        age_pred = self.age(user_embeds)
        occupation_pred = self.occupation(user_embeds)

        if (self.mode == 'gender'):
            loss_adv = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss_adv = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss_adv = fn_occupation(occupation_pred, occupation)

        return loss_adv, (age_pred, gender_pred, occupation_pred)

class NhopClassifier(nn.Module):
    def __init__(self, embed_dim, embeddings, edges, hop = 2):
        super(NhopClassifier, self).__init__()
        self.embeddings = embeddings
        self.mode = None
        self.hop = hop
        h = embed_dim

        self.age = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 7))

        self.occupation = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 21))

        self.gender = nn.Sequential(
            nn.Linear(h, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        edges_np = edges.numpy()
        edge_list = []
        for i in tqdm(range(edges_np.shape[1])):
            edge_list.append((edges_np[0,i], edges_np[1,i]))
        self.G = nx.Graph(edge_list)

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, features):
        (user, gender, occupation, age) = features

        neighbor = []

        user_np = list(user.cpu().numpy())

        k=0
        include_indices=[]
        ign_indices=[]
        ign_str = "failed vertices: "
        for user in user_np:
            final = user
            path = dict()
            for _ in range(self.hop):
                neighbor = list(self.G.neighbors(final))
                L = lambda a: len(a)
                orig_len = L(neighbor)

                cond = True
                cand = neighbor.pop(np.random.randint(0, L (neighbor)))
                if (cand in path):

                    # This is bounded by O(self.hop)
                    while(L(neighbor) > 0):
                        cand = neighbor.pop(np.random.randint(0, L (neighbor)))
                        if (cand not in path):
                            break
                    if (L(neighbor)==0):
                        ign_str+= "[{} <= {}]".format(orig_len, len(path))
                        cond = False
                        break
                            
                if cond:
                    path[cand]=0
                    final = cand
                else:
                    break
            if cond:
                neighbor_sub.append(final)
                include_indices.append(k)
            else:
                ign_indices.append(k)
            k+=1

        include_indices= np.array(include_indices)
        if (len(ign_indices)>0): print(ign_indices)
        #print(include_indices)
        gender= gender[include_indices]
        age = age[include_indices]
        occupation=occupation[include_indices]

        neighbor = neighbor_sub

        embeddings = self.embeddings[neighbor, :]

        age_pred = self.age(embeddings)
        gender_pred = self.gender(embeddings)
        occupation_pred = self.occupation(embeddings)

        fn_gender = nn.BCEWithLogitsLoss()
        fn_age = nn.CrossEntropyLoss()
        fn_occupation = nn.CrossEntropyLoss()

        if (self.mode == 'gender'):
            loss = fn_gender(gender_pred, gender.float())
        elif (self.mode == 'age'):
            loss = fn_age(age_pred, age)
        elif (self.mode == 'occupation'):
            loss = fn_occupation(occupation_pred, occupation)

        return loss, [age_pred, gender_pred, occupation_pred]
