"""
The authors combine multiple techniques like a learnable aggregation function
and residual layers to enable deeper GNN's

https://arxiv.org/pdf/2006.07739.pdf
"""

import torch
import torch.nn.functional as F

from torch.nn import Linear, LayerNorm, ReLU
from torch.nn.modules.activation import LogSoftmax
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
from torch_geometric.nn import GCNConv

class NLayerDeeperGCN(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes,
                    n_layers, aggr='softmax', layers_per_layer=2, norm_type='layer', dropout=0.1):
        super(NLayerDeeperGCN, self).__init__()
        self.n_layers = n_layers

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)
        
        self.convs = torch.nn.ModuleList()
        for i in range(1, n_layers):
            conv = GENConv(feature_size, feature_size, aggr=aggr,
                           t=1.0, learn_t=True, num_layers=layers_per_layer, norm=norm_type)
            norm = LayerNorm(feature_size, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout,
                                 ckpt_grad=i % 3)
            self.convs.append(layer)

        # I could not get the DeepGCN layer working yet to reduce to num_classes so simple layer for now
        self.convs.append(GCNConv(feature_size, num_classes))

    def forward(self, data):
        # NOTE: Deeper GCN does also not take edge weights
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x).squeeze()

        for i in range(self.n_layers-1):
            x = self.convs[i](x, edge_index)

        x = self.convs[self.n_layers-1](x, edge_index, edge_weight=edge_weights)

        return F.log_softmax(x, dim=1)

    def __str__(self):
        return str(self.n_layers) + "DeeperGCN"
    def __repr__(self):
        return str(self.n_layers) + "DeeperGCN"