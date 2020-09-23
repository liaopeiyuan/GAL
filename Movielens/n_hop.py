
# -*- coding: utf-8 -*-

from helper import Path, os, to_device, make_dataset_1M, create_optimizer, ltensor, collate_fn, node_cls_collate_fn, get_logger, train_node_cls, test_node_cls, train_gda, test_gda
from datasets import GDADataset, NodeClassification
from models import SharedBilinearDecoder, NhopClassifier, GNN, GAL_Nhop, NodeClassifier, NeighborClassifier

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

    train_gda_set = GDADataset(args.train_ratings, args.users_train, args.prefetch_to_gpu)
    test_gda_set = GDADataset(args.test_ratings, args.users_test, args.prefetch_to_gpu)

    train_fairness_set = NodeClassification(args.users_train, args.prefetch_to_gpu)
    test_fairness_set = NodeClassification(args.users_test, args.prefetch_to_gpu)

    edges = np.hstack((np.stack([args.train_ratings['user_id'].values,
                                 args.train_ratings['movie_id'].values]),
                       np.stack([args.train_ratings['movie_id'].values,
                                 args.train_ratings['user_id'].values])))
    edges = torch.LongTensor(edges)

    def get_model():
        decoder = SharedBilinearDecoder(args.num_rel, 2, args.embed_dim).to(args.device)
        model = GAL_Nhop(decoder, args.embed_dim, args.num_ent, edges, args, hop=args.hop).to(args.device)
        return model, decoder

    if args.prefetch_to_gpu:
        train_loader = DataLoader(train_gda_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_gda_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4, pin_memory=True, collate_fn=collate_fn)

    node_cls_loader = DataLoader(train_fairness_set, batch_size=256, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=True, collate_fn=node_cls_collate_fn)

    args.logger.info('Lambda: {}'.format(args.lambda_reg))

    model, decoder = get_model()

    optimizer_task = create_optimizer([
                                {'params': model.encoder.parameters()},
                                {'params': model.batchnorm.parameters()},
                                {'params': model.decoder.parameters()},
                                {'params':model.gnn.parameters()}], 'adam', args.lr)

    optimizer_adv_gender = create_optimizer([
                                {'params': model.encoder.parameters()},
                                {'params': model.batchnorm.parameters()},
                                {'params': model.gender.parameters()},
                                {'params':model.gnn.parameters()}],  'adam', args.lr * args.lambda_reg)

    args.logger.info('GDA for Gender Attribute')
    model.set_mode('gender')

    for epoch in tqdm(range(args.num_epochs)):

        if epoch % (args.valid_freq) == 0 and epoch >= 15:
            with torch.no_grad():
                test_gda(test_gda_set, args, model)

        train_gda(train_loader, node_cls_loader,args, model, optimizer_task, optimizer_adv_gender, False)
        gc.collect()

    embeddings = model.encode(None).detach().squeeze(0)
    attacker = NeighborClassifier(args.embed_dim, embeddings, edges).cuda() #NhopClassifier(args.embed_dim, embeddings, edges, hop=args.hop).cuda()
    optimizer_attacker_gender = create_optimizer(attacker.gender.parameters(), 'adam', args.lr)

    args.logger.info('Gender Neighbor Adversary')
   
    attacker.set_mode('gender')
    for epoch in tqdm(range(1, args.finetune_epochs)):
        train_node_cls(node_cls_loader, args, attacker, optimizer_attacker_gender)
        gc.collect()
        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, attacker, mode='gender')

    dirname = os.path.join('./checkpoints', args.experiment, args.task, args.model)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    logname = args.logname

    path = (os.path.join(dirname,logname+"model_gender.pth"))
    torch.save(model.state_dict(), path)
    path = (os.path.join(dirname,logname+"attacker_neighbor_gender.pth"))
    torch.save(attacker.state_dict(), path)


    embeddings = model.encode(None).detach().squeeze(0)
    attacker = NodeClassifier(args.embed_dim, embeddings).cuda()
    optimizer_attacker_gender = create_optimizer(attacker.gender.parameters(), 'adam', args.lr)

    args.logger.info('Gender Node Adversary')
   
    attacker.set_mode('gender')
    for epoch in tqdm(range(args.finetune_epochs)):
        train_node_cls(node_cls_loader, args, attacker, optimizer_attacker_gender)
        gc.collect()
        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, attacker, mode='gender')

    path = (os.path.join(dirname,logname+"attacker_node_gender.pth"))
    torch.save(attacker.state_dict(), path)

    model, decoder = get_model()

    optimizer_task = create_optimizer([
                                {'params': model.encoder.parameters()},
                                {'params': model.batchnorm.parameters()},
                                {'params': model.decoder.parameters()}, 
                                {'params':model.gnn.parameters()}], 'adam', args.lr)

    optimizer_adv_age = create_optimizer([
                                {'params': model.encoder.parameters()},
                                {'params': model.batchnorm.parameters()},
                                {'params': model.age.parameters()},
                                {'params':model.gnn.parameters()}], 'adam', args.lr * args.lambda_reg)

    args.logger.info('GDA for Age Attribute')
    model.set_mode('age')

    for epoch in tqdm(range(args.num_epochs)):

        if epoch % (args.valid_freq) == 0 and epoch >= 15:
            with torch.no_grad(): 
                test_gda(test_gda_set, args, model)

        train_gda(train_loader, node_cls_loader,args, model, optimizer_task, optimizer_adv_age, False)
        gc.collect()

    embeddings = model.encode(None).detach().squeeze(0)
    attacker = NeighborClassifier(args.embed_dim, embeddings, edges).cuda() #NhopClassifier(args.embed_dim, embeddings, edges, hop=args.hop).cuda()
    optimizer_attacker_age = create_optimizer(attacker.age.parameters() ,'adam', args.lr)

    args.logger.info('Age Neighbor Adversary')
   

    attacker.set_mode('age')
    for epoch in tqdm(range(args.finetune_epochs)):
        train_node_cls(node_cls_loader, args, attacker, optimizer_attacker_age)
        gc.collect()
        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, attacker, mode='age')

    path = (os.path.join(dirname,logname+"model_age.pth"))
    torch.save(model.state_dict(), path)
    path = (os.path.join(dirname,logname+"attacker_neighbor_age.pth"))
    torch.save(attacker.state_dict(), path)

    embeddings = model.encode(None).detach().squeeze(0)
    attacker = NodeClassifier(args.embed_dim, embeddings).cuda()
    optimizer_attacker_age = create_optimizer(attacker.age.parameters() ,'adam', args.lr)

    args.logger.info('Age Node Adversary')
   

    attacker.set_mode('age')
    for epoch in tqdm(range(args.finetune_epochs)):
        train_node_cls(node_cls_loader, args, attacker, optimizer_attacker_age)
        gc.collect()
        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, attacker, mode='age')

    path = (os.path.join(dirname,logname+"attacker_node_age.pth"))
    torch.save(attacker.state_dict(), path)

    model, decoder = get_model()

    optimizer_task = create_optimizer([
                                {'params': model.encoder.parameters()},
                                {'params': model.batchnorm.parameters()},
                                {'params': model.decoder.parameters()},
                                {'params':model.gnn.parameters()}], 'adam', args.lr)

    optimizer_adv_occupation = create_optimizer([
                                {'params': model.encoder.parameters()},
                                {'params': model.batchnorm.parameters()},
                                {'params': model.occupation.parameters()},
                                {'params':model.gnn.parameters()}], 'adam', args.lr * args.lambda_reg)


    args.logger.info('GDA for Occupation Attribute')
    model.set_mode('occupation')

    for epoch in tqdm(range(args.num_epochs)):

        if epoch % (args.valid_freq) == 0 and epoch >= 15:
            with torch.no_grad():
                test_gda(test_gda_set, args, model)

        train_gda(train_loader, node_cls_loader,args, model, optimizer_task, optimizer_adv_occupation, False)
        gc.collect()

    embeddings = model.encode(None).detach().squeeze(0)
    attacker = NeighborClassifier(args.embed_dim, embeddings, edges).cuda() #NhopClassifier(args.embed_dim, embeddings, edges, hop=args.hop).cuda()
    optimizer_attacker_occupation = create_optimizer(attacker.occupation.parameters(), 'adam', args.lr) 

    args.logger.info('Occupation Neighbor Adversary')
   
    attacker.set_mode('occupation')
    for epoch in tqdm(range(args.finetune_epochs)):
        train_node_cls(node_cls_loader, args, attacker, optimizer_attacker_occupation)
        gc.collect()
        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, attacker, mode='occupation')

    path = (os.path.join(dirname,logname+"model_occupation.pth"))
    torch.save(model.state_dict(), path)
    path = (os.path.join(dirname,logname+"attacker_neighbor_occupation.pth"))
    torch.save(attacker.state_dict(), path)

    embeddings = model.encode(None).detach().squeeze(0)
    attacker = NodeClassifier(args.embed_dim, embeddings).cuda()
    optimizer_attacker_occupation = create_optimizer(attacker.occupation.parameters(), 'adam', args.lr) 

    args.logger.info('Occupation Node Adversary')
   
    attacker.set_mode('occupation')
    for epoch in tqdm(range(args.finetune_epochs)):
        train_node_cls(node_cls_loader, args, attacker, optimizer_attacker_occupation)
        gc.collect()
        with torch.no_grad():
            rmse, test_loss = test_node_cls(test_fairness_set, args, attacker, mode='occupation')

    path = (os.path.join(dirname,logname+"attacker_node_occupation.pth"))
    torch.save(attacker.state_dict(), path)

if __name__ == '__main__':
    assert(False) # You shouldn't run this. Please call exec.py 
