"""
In the paper "Simplifying Graph Convolutional Networks" the authors
propose that the non-linear activation layer between GCN layers are 
redundant.

https://arxiv.org/pdf/1902.07153.pdf
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv

class SimpleGCN(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes,
                    hops=2):
        super(SimpleGCN, self).__init__()
        self.hops = hops

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)
        
        self.conv1 = SGConv(feature_size, num_classes, K=hops)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x).squeeze()

        x = self.conv1(x, edge_index, edge_weight=edge_weights)

        return F.log_softmax(x, dim=1)

    def __str__(self):
        return "SimpleGCN (k=" + str(self.hops) + ")"