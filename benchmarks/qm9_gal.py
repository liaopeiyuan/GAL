import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
import argparse
import random
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
parser.add_argument('--seed', type=int, default=50, help='random seed')
parser.add_argument('--lambda_reg', type=float, default=0, help='Regularization strength for gradient ascent-descent')
parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
parser.add_argument('--gnn_types', type=str, default='ChebConv', help='Types of GNN')
parser.add_argument('--finetune_epochs', type=int, default=30, help='Finetune epochs')
parser.add_argument('--target', type=int, default=0, help='target')
parser.add_argument('--target_attack', type=int, default=1, help='target_attack')
parser.add_argument('--hidden_dim', type=int, default=64, help='target_attack')
parser.add_argument('--batch_size', type=int, default=128, help='target_attack')

args = parser.parse_args()

result = {}

for seed in [500,600,700,800,900]:
    result[seed] = {}

    args.seed = seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    target = args.target
    target_attack = args.target_attack
    dim = args.hidden_dim
    bsize = args.batch_size

    class MyTransform(object):
        def __call__(self, data):
            return data


    class Complete(object):
        def __call__(self, data):
            device = data.edge_index.device

            row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
            col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

            row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
            col = col.repeat(data.num_nodes)
            edge_index = torch.stack([row, col], dim=0)

            edge_attr = None
            if data.edge_attr is not None:
                idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
                size = list(data.edge_attr.size())
                size[0] = data.num_nodes * data.num_nodes
                edge_attr = data.edge_attr.new_zeros(size)
                edge_attr[idx] = data.edge_attr

            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            data.edge_attr = edge_attr
            data.edge_index = edge_index

            return data


    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform).shuffle()

    print(dataset.data.x.size())

    print(dataset.data.y.size())
    print(dataset.data.edge_attr.size())

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    print(mean.size())
    print(std.size())

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:]
    test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=bsize, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=True, num_workers=4)

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
        def __init__(self):
            super(Net, self).__init__()
            self.lin0 = torch.nn.Linear(dataset.num_features, dim)

            nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
            self.conv = NNConv(dim, dim, nn, aggr='mean')
            self.gru = GRU(dim, dim)
            self.set2set = Set2Set(dim, processing_steps=3)

            self.lin1 = torch.nn.Linear(2 * dim, dim)
            self.lin2 = torch.nn.Linear(dim, 1)

            self.attacker = torch.nn.Sequential(
                torch.nn.Linear(2 * dim, dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, 1)
            )

            self.attr = torch.nn.Sequential(
                GradientReversalLayer(),
                torch.nn.Linear(2 * dim, dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, 1)
            )


        def forward(self, data, a = False):
            out = F.relu(self.lin0(data.x))
            h = out.unsqueeze(0)

            for i in range(3):
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)

            out = self.set2set(out, data.batch)

            if a:
                res = self.attacker(out).view(-1)
            else:
                res = self.attr(out).view(-1)

            out = F.relu(self.lin1(out))
            out = self.lin2(out)

            return out.view(-1), res

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv.parameters(), weight_decay=0),
        dict(params=model.gru.parameters(), weight_decay=0),
        dict(params=model.set2set.parameters(), weight_decay=0),
        dict(params=model.lin1.parameters(), weight_decay=0),
        dict(params=model.lin2.parameters(), weight_decay=0),
    ], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.7, patience=5,
                                                        min_lr=0.00001)

    optimizer_attr = torch.optim.Adam([
        dict(params=model.conv.parameters(), weight_decay=0),
        dict(params=model.gru.parameters(), weight_decay=0),
        dict(params=model.set2set.parameters(), weight_decay=0),
        dict(params=model.attr.parameters(), weight_decay=0),
    ], lr=args.lr*args.lambda_reg)

    def train(epoch):
        model.train()
        loss_all = 0

        for _, data in enumerate(tqdm(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            tar, att = model(data, a=False)
            loss = F.mse_loss(tar, data.y[:,target])

            loss.backward(retain_graph=True)
            loss_all += loss.item() * data.num_graphs
            optimizer.step()


            optimizer_attr.zero_grad()
            loss2 = F.mse_loss(att, data.y[:,target_attack])
            loss2.backward()
            optimizer_attr.step()

        return loss_all / len(train_loader.dataset)


    def test(loader):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            s = std[0,target]

            error += (model(data)[0] * s - data.y[:,target] * s).abs().sum().item()  # MAE
        return error / len(loader.dataset)

    result[seed]['task'] = {}

    best_val_error = None
    for epoch in range(1, args.num_epochs+1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        try:
            result[seed]['task']['test'].append(test_error)
        except:
            result[seed]['task']['test']=[test_error]

        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
            'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

    result[seed]['task']['val'] = best_val_error

    optimizer_att = torch.optim.Adam([
        dict(params=model.attacker.parameters(), weight_decay=0),
    ], lr=args.lr)


    def train_attr(epoch):
        model.train()
        loss_all = 0

        for data in train_loader:
            data = data.to(device)
            optimizer_att.zero_grad()
            tar, att = model(data, a=True)
            loss = F.mse_loss(att, data.y[:,target_attack])
                    
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer_att.step()
        return loss_all / len(train_loader.dataset)


    def test_attr(loader):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            s = std[0, target_attack]
            error += (model(data, a=True)[1] * s - data.y[:,target_attack] * s).abs().sum().item()  # MAE
        return error / len(loader.dataset)

    result[seed]['adv'] = {}

    best_val_error = None
    for epoch in range(1, args.finetune_epochs+1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train_attr(epoch)
        val_error = test_attr(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test_attr(test_loader)
            best_val_error = val_error

        print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
            'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))

        try:
            result[seed]['adv']['test'].append(test_error)
        except:
            result[seed]['adv']['test']=[test_error]

    result[seed]['adv']['val'] = best_val_error

print(result)