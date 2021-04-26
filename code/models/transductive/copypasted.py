import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes):
        super(GAT, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)

        self.conv1 = GATConv(feature_size, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.embedding(x).squeeze()

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=-1)

    def __str__(self):
        return "TransductiveGAT"

from torch_geometric.nn import SuperGATConv

class SuperGAT(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes):
        super(SuperGAT, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)

        self.conv1 = SuperGATConv(feature_size, 8, heads=8,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.conv2 = SuperGATConv(8 * 8, num_classes, heads=8,
                                  concat=False, dropout=0.6,
                                  attention_type='MX', edge_sample_ratio=0.8,
                                  is_undirected=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.embedding(x).squeeze()

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        return F.log_softmax(x, dim=-1)

    def __str__(self):
        return "TransductiveSuperGAT"

import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import AGNNConv


class AGNN(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes):
        super(AGNN, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)

        self.lin1 = torch.nn.Linear(feature_size, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.embedding(x).squeeze()

        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

    def __str__(self):
        return "TransductiveAGNN"
