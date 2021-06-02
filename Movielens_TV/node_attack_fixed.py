
# -*- coding: utf-8 -*-

from helper import os, Path, to_device, make_dataset_1M, create_optimizer, ltensor, collate_fn, node_cls_collate_fn, get_logger, train_node_cls, test_node_cls, train_gda, test_gda
from helper import train_gcmc, test_gcmc 
from datasets import GDADataset, NodeClassification, KBDataset
from models import SharedBilinearDecoder, NodeClassifier, GNN, SimpleGCMC

from tqdm import tqdm
import numpy as np
import torch
import itertools
import gc
import numpy as np
from torch.utils.data import Dataset, DataLoader

def run(args):
    args.train_ratings.to_csv('train_ratings_ml.csv')
    args.test_ratings.to_csv('test_ratings_ml.csv')

    train_set = KBDataset(args.train_ratings, args.prefetch_to_gpu)
    test_set = KBDataset(args.test_ratings, args.prefetch_to_gpu)

    train_fairness_set = NodeClassification(args.users_train, args.prefetch_to_gpu)
    test_fairness_set = NodeClassification(args.users_test, args.prefetch_to_gpu)

    edges = np.hstack((np.stack([args.train_ratings['user_id'].values,
                                 args.train_ratings['movie_id'].values]),
                       np.stack([args.train_ratings['movie_id'].values,
                                 args.train_ratings['user_id'].values])))
    edges = torch.LongTensor(edges)

    def get_model():
        decoder = SharedBilinearDecoder(args.num_rel, 2, args.embed_dim).to(args.device)
        model = SimpleGCMC(decoder, args.embed_dim, args.num_ent).to(args.device)
        return model, decoder

    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)

    node_cls_loader = DataLoader(train_fairness_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=4, pin_memory=True, collate_fn=node_cls_collate_fn)


    model, decoder = get_model()

    optimizer = create_optimizer(model.parameters(), 'adam', args.lr)

    for epoch in tqdm(range(args.num_epochs)):

        if epoch % args.valid_freq == 0:
            with torch.no_grad():
                rmse, test_loss = test_gcmc(test_set, args, model)

        train_gcmc(train_loader, epoch, args, model, optimizer)
        gc.collect()

        if epoch % (args.valid_freq * 5) == 0:
            rmse = test_gcmc(test_set, args, model)

    def freeze_network(model):
        for name, p in model.named_parameters():
            p.requires_grad = False

    embeddings = model.encode(None).detach().squeeze(0)
    model_cls = NodeClassifier(args.embed_dim,embeddings).cuda()
    optimizer_gender = create_optimizer(model_cls.gender.parameters(), 'adam', args.lr)
    optimizer_age = create_optimizer(model_cls.age.parameters(), 'adam', args.lr)
    optimizer_occupation = create_optimizer(model_cls.occupation.parameters(), 'adam', args.lr)

   
    model_cls.set_mode('gender')
    for epoch in tqdm(range(args.finetune_epochs)):

        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, model_cls, mode='gender')

        train_node_cls(node_cls_loader, args, model_cls, optimizer_gender)
        gc.collect()

    model_cls.set_mode('age')
    for epoch in tqdm(range(args.finetune_epochs)):

        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, model_cls, mode='age')

        train_node_cls(node_cls_loader, args, model_cls, optimizer_age)
        gc.collect()

    model_cls.set_mode('occupation')
    for epoch in tqdm(range(args.finetune_epochs)):

        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, model_cls, mode='occupation')

        train_node_cls(node_cls_loader, args, model_cls, optimizer_occupation)
        gc.collect()


if __name__ == '__main__':
    assert(False) # You shouldn't run this. Please call exec.py 
