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

tqdm.monitor_interval = 0
sys.path.append('../')

def to_device(tensor):
    if tensor is not None: return tensor.to("cuda")

def make_dataset_1M(load_sidechannel=False):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('../data/ml-1m/ratings.dat', sep='::', names=r_cols,
                          encoding='latin-1',engine='python')
    shuffled_ratings = ratings.sample(frac=1).reset_index(drop=True)
    train_cutoff_row = int(np.round(len(shuffled_ratings) * 0.9))
    train_ratings = shuffled_ratings[:train_cutoff_row]
    test_ratings = shuffled_ratings[train_cutoff_row:]
    if load_sidechannel:
        u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
        m_cols = ['movie_id', 'title', 'genre']
        users = pd.read_csv('../data/ml-1m/users.dat', sep='::', names=u_cols,
                            encoding='latin-1', parse_dates=True,engine='python')
        movies = pd.read_csv('../data/ml-1m/movies.dat', sep='::', names=m_cols,
                             encoding='latin-1', parse_dates=True,engine='python')

    train_ratings.drop("unix_timestamp", inplace=True, axis=1)
    train_ratings_matrix = train_ratings.pivot_table(index=['movie_id'], \
                                                     columns=['user_id'], values='rating').reset_index(drop=True)
    test_ratings.drop("unix_timestamp", inplace=True, axis=1)
    columnsTitles = ["user_id", "rating", "movie_id"]
    train_ratings = train_ratings.reindex(columns=columnsTitles) - 1
    test_ratings = test_ratings.reindex(columns=columnsTitles) - 1
    users.user_id = users.user_id.astype(np.int64)
    movies.movie_id = movies.movie_id.astype(np.int64)
    users['user_id'] = users['user_id'] - 1
    movies['movie_id'] = movies['movie_id'] - 1

    if load_sidechannel:
        return train_ratings, test_ratings, users, movies
    else:
        return train_ratings, test_ratings

def create_optimizer(params, mode, *args, **kwargs):
    if mode == 'SGD':
        opt = optim.SGD(params, *args, momentum=0., **kwargs)
    elif mode.startswith('nesterov'):
        momentum = float(mode[len('nesterov'):])
        opt = optim.SGD(params, *args, momentum=momentum, nesterov=True, **kwargs)
    elif mode.lower() == 'adam':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True,
                         weight_decay=1e-4, **kwargs)
    elif mode.lower() == 'adam_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_hyp3':
        betas = kwargs.pop('betas', (0., .99))
        opt = optim.Adam(params, *args, betas=betas, amsgrad=True, **kwargs)
    elif mode.lower() == 'adam_sparse':
        betas = kwargs.pop('betas', (.9, .999))
        opt = optim.SparseAdam(params, *args, weight_decay=1e-4, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp2':
        betas = kwargs.pop('betas', (.5, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == 'adam_sparse_hyp3':
        betas = kwargs.pop('betas', (.0, .99))
        opt = optim.SparseAdam(params, *args, betas=betas)
    elif mode.lower() == 'adabound':
        opt = adabound.AdaBound(params, *args, final_lr=0.1)
    else:
        raise NotImplementedError()
    return opt

ltensor = torch.LongTensor

def collate_fn(batch):
    if isinstance(batch, np.ndarray) or (isinstance(batch, list) and isinstance(batch[0], np.ndarray)):
        return ltensor(batch).contiguous()
    else:
        return torch.stack(batch).contiguous()  

def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read
    
    Returns
    -------
    A logger object which writes to both file and stdout
        
    """
    config_dict = json.load(open( config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir, name.replace('/', '-'))
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def node_cls_collate_fn(batch):
    users = []
    genders = []
    occupations = []
    ages = []
    for [user, gender, occupation, age] in batch:
        users.append(user)
        genders.append(gender)
        occupations.append(occupation)
        ages.append(age)

    users = ltensor(users)
    genders = ltensor(genders)
    occupations = ltensor(occupations)
    ages = ltensor(ages)

    return (users, genders, occupations, ages)


def train_node_cls(data_loader, args, model, optimizer):
    model.train()
    if args.show_tqdm:
        data_itr = tqdm(enumerate(data_loader))
    else:
        data_itr = enumerate(data_loader)

    for idx, (user, gender, occupation, age) in data_itr:

        if args.use_cuda:
            (user, gender, occupation, age) = (user.cuda(), gender.cuda(), occupation.cuda(), age.cuda())

        task_loss, preds = model((user, gender, occupation, age))
        optimizer.zero_grad()
        full_loss = task_loss
        full_loss.backward()

        optimizer.step()


def test_node_cls(test_fairness_set, args, model, mode='age'):
    model.eval()
    node_cls_test_loader = DataLoader(test_fairness_set, batch_size=4000, shuffle=False,
                                      drop_last=False,
                                      num_workers=1, pin_memory=True, collate_fn=node_cls_collate_fn)

    for idx, (user, gender, occupation, age) in tqdm(enumerate(node_cls_test_loader)):
        (user, gender, occupation, age) = (user.cuda(), gender.cuda(), occupation.cuda(), age.cuda())
        task_loss, [pred_age, pred_gender, pred_occupation] = model((user, gender, occupation, age))


        pred_age = pred_age.max(1)[1]
        pred_occupation = pred_occupation.max(1)[1]
        pred_gender = (pred_gender > 0.5)

        to_np = lambda x: x.detach().cpu().numpy()
        pred_age, truth_age = to_np(pred_age), to_np(age)
        pred_occupation, truth_occupation = to_np(pred_occupation), to_np(occupation)
        pred_gender, truth_gender = to_np(pred_gender), to_np(gender)

        macro_gender = f1_score(pred_gender, truth_gender, average='macro') if mode =='gender' else 0
        macro_age = f1_score(pred_age, truth_age, average='macro') if mode =='age' else 0
        macro_occupation = f1_score(pred_occupation, truth_occupation, average='macro') if mode =='occupation' else 0

        roc_auc = roc_auc_score(truth_gender, pred_gender)
        if mode =='gender':
            conf = confusion_matrix(truth_gender, pred_gender)
        elif mode == 'age':
            conf = confusion_matrix(truth_age, pred_age)
        elif mode == 'occupation':
            conf = confusion_matrix(truth_occupation, pred_occupation)

        args.logger.info("Confusion Matrix\n"+str(conf))

        log = 'Macro F1/AUC: Gender: {:.4f}/{:.4f} Age: {:.4f} Occupation: {:.4f}\n===================='
        args.logger.info(log.format(macro_gender, roc_auc, macro_age, macro_occupation))

    rms, test_loss = 0,0
    return rms, test_loss

def train_gda(data_loader, adv_loader, args, model, optimizer_task, optimizer_adv, pretrain=False):
    model.train()
    adv_loader = itertools.cycle(adv_loader)

    if args.show_tqdm:
        data_itr = tqdm(enumerate(zip(data_loader, adv_loader)))
    else:
        data_itr = enumerate(zip(data_loader, adv_loader))

    for idx, (p_batch ,(user, gender, occupation, age)) in data_itr:

        if args.use_cuda:
            p_batch = p_batch.cuda()
            (user, gender, occupation, age) = (user.cuda(), gender.cuda(), occupation.cuda(), age.cuda())
            
        loss_task, preds_task = model(p_batch)

        if True:
            optimizer_task.zero_grad()
            loss_task.backward(retain_graph=True)
            optimizer_task.step()
            optimizer_task.zero_grad()

        if not(pretrain):
            loss_adv, (age_pred, gender_pred, occupation_pred) = model.forward_attr((user, gender, occupation, age))
            optimizer_adv.zero_grad()
            loss_adv.backward(retain_graph=True)
            optimizer_adv.step()
            optimizer_adv.zero_grad()

def test_gda(dataset, args, model):
    test_loader = DataLoader(dataset, batch_size=4000, num_workers=1, collate_fn=collate_fn)
    cst_inds = np.arange(args.num_ent, dtype=np.int64)[:, None]
    if args.show_tqdm:
        data_itr = tqdm(enumerate(test_loader))
    else:
        data_itr = enumerate(test_loader)

    (user, gender, occupation, age) = dataset.user_features
    (user, gender, occupation, age) = (user.cuda(), gender.cuda(), occupation.cuda(), age.cuda())

    preds_list = []
    rels_list = []
    for idx, p_batch in data_itr:
        p_batch = (p_batch).cuda()
        lhs, rel, rhs = p_batch[:, 0], p_batch[:, 1], p_batch[:, 2]
        loss_task, preds = model(p_batch)
        loss_adv, (age_pred, gender_pred, occupation_pred) = model.forward_attr((user, gender, occupation, age))
        rel += 1
        preds_list.append(preds.squeeze())
        rels_list.append(rel.float())
    total_preds = torch.cat(preds_list)
    total_rels = torch.cat(rels_list)

    predictions = total_preds.round().detach().cpu().numpy()

    rms = torch.sqrt(F.mse_loss(total_preds.squeeze(), total_rels.squeeze()))
    args.logger.info("Adversarial Loss: {}".format(loss_adv.item()))
    args.logger.info("Edge RMSE:  {}".format(rms.item()))

    return

def train_gcmc(data_loader, counter, args, modelD, optimizer):
    if args.show_tqdm:
        data_itr = tqdm(enumerate(data_loader))
    else:
        data_itr = enumerate(data_loader)

    for idx, p_batch in data_itr:

        if args.use_cuda:
            p_batch = p_batch.cuda()

        p_batch_var = (p_batch)

        task_loss, preds = modelD(p_batch_var)
        optimizer.zero_grad()
        full_loss = task_loss
        full_loss.backward(retain_graph=False)
        optimizer.step()

def test_gcmc(dataset, args, modelD):
    test_loader = DataLoader(dataset, batch_size=4000, num_workers=1, collate_fn=collate_fn)
    cst_inds = np.arange(args.num_ent, dtype=np.int64)[:, None]
    if args.show_tqdm:
        data_itr = tqdm(enumerate(test_loader))
    else:
        data_itr = enumerate(test_loader)

    preds_list = []
    rels_list = []
    test_loss_list = []
    for idx, p_batch in data_itr:
        p_batch_var = (p_batch).cuda()
        lhs, rel, rhs = p_batch_var[:, 0], p_batch_var[:, 1], p_batch_var[:, 2]
        test_loss, preds = modelD(p_batch_var)
        rel += 1
        preds_list.append(preds.squeeze())
        rels_list.append(rel.float())
        test_loss_list.append(test_loss)
    total_preds = torch.cat(preds_list)
    total_rels = torch.cat(rels_list)
    test_loss = torch.mean(torch.stack(test_loss_list))

    predictions = total_preds.round().detach().cpu().numpy()
    args.logger.info("Confusion Matrix\n"+str(confusion_matrix(total_rels.detach().cpu().numpy(), predictions)))

    rms = torch.sqrt(F.mse_loss(total_preds.squeeze(), total_rels.squeeze()))
    args.logger.info("Test RMSE: {}".format(rms.item()))
    return rms, test_loss
