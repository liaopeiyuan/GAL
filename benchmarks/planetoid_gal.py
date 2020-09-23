import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GINConv, GATConv  # noqa
from torch_geometric.utils import train_test_split_edges
import argparse
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score, f1_score
import json
from torch.nn import Sequential, ReLU, Linear

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

class GradientReversalLayer(torch.nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, inputs):
        return GradReverse.apply(inputs)

class Net(torch.nn.Module):
    def __init__(self, name='GCNConv'):
        super(Net, self).__init__()
        self.name = name
        if (name == 'GCNConv'):
            self.conv1 = GCNConv(dataset.num_features, 128)
            self.conv2 = GCNConv(128, 64)
        elif (name == 'ChebConv'):
            self.conv1 = ChebConv(dataset.num_features, 128, K=2)
            self.conv2 = ChebConv(128, 64, K=2)
        elif (name == 'GATConv'):
            self.conv1 = GATConv(dataset.num_features, 128)
            self.conv2 = GATConv(128, 64)
        elif (name == 'GINConv'):
            nn1 = Sequential(Linear(dataset.num_features, 128), ReLU(), Linear(128, 64))
            self.conv1 = GINConv(nn1)
            self.bn1 = torch.nn.BatchNorm1d(64)
            nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
            self.conv2 = GINConv(nn2)
            self.bn2 = torch.nn.BatchNorm1d(64)

        self.attr = GCNConv(64, dataset.num_classes, cached=True,
                                normalize=not args.use_gdc)

        self.attack = GCNConv(64, dataset.num_classes, cached=True,
                            normalize=not args.use_gdc)
        self.reverse = GradientReversalLayer()

    def forward(self, pos_edge_index, neg_edge_index):

        if (self.name == 'GINConv'):
            x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
            x = self.bn1(x)
            x = F.relu(self.conv2(x, data.train_pos_edge_index))
            x = self.bn2(x)
        else:
            x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
            x = self.conv2(x, data.train_pos_edge_index)

        attr = self.attr(x, edge_index, edge_weight)

        
        attack = self.reverse(x)
        att = self.attack(attack, edge_index, edge_weight)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])

        res = torch.einsum("ef,ef->e", x_i, x_j)

        return res, F.log_softmax(attr, dim=1), F.log_softmax(att, dim=1)

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--num_epochs', type=int, default=175, help='Number of training epochs (default: 500)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=30, help='random seed')
parser.add_argument('--lambda_reg', type=float, default=1, help='Regularization strength for gradient ascent-descent')
parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
parser.add_argument('--gnn_types', type=str, default='ChebConv', help='Types of GNN')
parser.add_argument('--finetune_epochs', type=int, default=80, help='Finetune epochs')
parser.add_argument('--dataset', type=str, default='Pubmed', help='dataset')


args = parser.parse_args()

res = {}
for seed in [100,200,300,400,500]:
    for l in [0, 1.25, 1.75]:
        for m in ['GINConv','GCNConv','GATConv', 'ChebConv']:

            args.seed = seed
            args.lambda_reg = l

            try:
                res[m][seed][l]={}
            except:
                try:
                    res[m][seed] = {l:{}}
                except:
                    res[m] = {seed: {l:{}}}

            
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(args.seed)

            dataset = args.dataset
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
            dataset = Planetoid(path, dataset, T.NormalizeFeatures())
            data = dataset[0]

            if args.use_gdc:
                gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                            normalization_out='col',
                            diffusion_kwargs=dict(method='ppr', alpha=0.05),
                            sparsification_kwargs=dict(method='topk', k=128,
                                                    dim=0), exact=True)
                data = gdc(data)

            labels = data.y.cuda()
            edge_index, edge_weight = data.edge_index.cuda(), data.edge_attr

            print(labels.size())
            # Train/validation/test
            data = train_test_split_edges(data)

            print(labels)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, data = Net(m).to(device), data.to(device)
            
            if (m=='GINConv'):
                optimizer = torch.optim.Adam([
                    dict(params=model.conv1.parameters(), weight_decay=0),
                    dict(params=model.bn1.parameters(), weight_decay=0),
                    dict(params=model.conv2.parameters(), weight_decay=0),
                    dict(params=model.bn2.parameters(), weight_decay=0),
                ], lr=args.lr)
            else:
                optimizer = torch.optim.Adam([
                    dict(params=model.conv1.parameters(), weight_decay=0),
                    dict(params=model.conv2.parameters(), weight_decay=0)
                ], lr=args.lr)

            if (m=='GINConv'):
                optimizer_att = torch.optim.Adam([
                    dict(params=model.conv2.parameters(), weight_decay=5e-4), 
                    dict(params=model.bn2.parameters(), weight_decay=0),  
                    dict(params=model.attack.parameters(), weight_decay=5e-4),
                ], lr=args.lr * args.lambda_reg)
            else:
                optimizer_att = torch.optim.Adam([
                    dict(params=model.conv2.parameters(), weight_decay=5e-4),   
                    dict(params=model.attack.parameters(), weight_decay=5e-4),
                ], lr=args.lr * args.lambda_reg)

            def get_link_labels(pos_edge_index, neg_edge_index):
                link_labels = torch.zeros(pos_edge_index.size(1) +
                                        neg_edge_index.size(1)).float().to(device)
                link_labels[:pos_edge_index.size(1)] = 1.
                return link_labels


            def train():
                model.train()
                optimizer.zero_grad()

                x, pos_edge_index = data.x, data.train_pos_edge_index

                _edge_index, _ = remove_self_loops(pos_edge_index)
                pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                                num_nodes=x.size(0))

                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                    num_neg_samples=pos_edge_index.size(1))

                link_logits, attr_prediction, attack_prediction = model(pos_edge_index, neg_edge_index)
                link_labels = get_link_labels(pos_edge_index, neg_edge_index)

                loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                optimizer_att.zero_grad()
                loss2 = F.nll_loss(attack_prediction, labels)
                loss2.backward()
                optimizer_att.step()

                return loss


            def test():
                model.eval()
                perfs = []
                for prefix in ["val", "test"]:
                    pos_edge_index, neg_edge_index = [
                        index for _, index in data("{}_pos_edge_index".format(prefix),
                                                "{}_neg_edge_index".format(prefix))
                    ]
                    link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index)[0])
                    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
                    link_probs = link_probs.detach().cpu().numpy()
                    link_labels = link_labels.detach().cpu().numpy()
                    perfs.append(roc_auc_score(link_labels, link_probs))
                return perfs


            best_val_perf = test_perf = 0
            for epoch in range(1, args.num_epochs+1):
                train_loss = train()
                val_perf, tmp_test_perf = test()
                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    test_perf = tmp_test_perf
                    res[m][seed][l]['task'] = {'val':best_val_perf, 'test':test_perf}
                log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, train_loss, val_perf, tmp_test_perf))


            optimizer_attr = torch.optim.Adam([
                dict(params=model.attr.parameters(), weight_decay=5e-4),
            ], lr=args.lr)

            def train_attr():
                model.train()
                optimizer_attr.zero_grad()

                x, pos_edge_index = data.x, data.train_pos_edge_index

                _edge_index, _ = remove_self_loops(pos_edge_index)
                pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                                num_nodes=x.size(0))

                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                    num_neg_samples=pos_edge_index.size(1))

                F.nll_loss(model(pos_edge_index, neg_edge_index)[1][data.train_mask], labels[data.train_mask]).backward()
                optimizer_attr.step()


            @torch.no_grad()
            def test_attr():
                model.eval()
                accs = []
                m = ['train_mask', 'val_mask', 'test_mask']
                i = 0
                for _, mask in data('train_mask', 'val_mask', 'test_mask'):

                    if (m[i] == 'train_mask') :
                        x, pos_edge_index = data.x, data.train_pos_edge_index

                        _edge_index, _ = remove_self_loops(pos_edge_index)
                        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                                        num_nodes=x.size(0))

                        neg_edge_index = negative_sampling(
                            edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
                            num_neg_samples=pos_edge_index.size(1))
                    else:
                        pos_edge_index, neg_edge_index = [
                        index for _, index in data("{}_pos_edge_index".format(m[i].split("_")[0]),
                                                "{}_neg_edge_index".format(m[i].split("_")[0]))
                        ]
                    _, logits, _ = model(pos_edge_index, neg_edge_index)

                    pred = logits[mask].max(1)[1]

                    macro = f1_score((data.y[mask]).cpu().numpy(), pred.cpu().numpy(),average='macro')
                    accs.append(macro)

                    i+=1
                return accs


            best_val_acc = test_acc = 0
            for epoch in range(1, args.finetune_epochs+1):
                train_attr()
                train_acc, val_acc, tmp_test_acc = test_attr()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    res[m][seed][l]['adversary'] = {'val':best_val_acc, 'test':test_acc}
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, train_acc, val_acc, tmp_test_acc))

print(res)