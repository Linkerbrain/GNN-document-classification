"""
Petar proposes to add a mechanism that weights the contribution of a node to
another node based on both nodes

https://arxiv.org/pdf/1710.10903.pdf
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class NLayerGat(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes,
                    n_layers, n_heads, dropout, concat):
        super(NLayerGat, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)
        
        self.convs = torch.nn.ModuleList()

        # If the heads are concattenated, the feature size is still kept together
        # so make sure the n_heads can evenly divide feature_size
        output_per_head = feature_size // n_heads if concat else feature_size

        for i in range(n_layers-1):
            self.convs.append(GATConv(feature_size, output_per_head, heads=n_heads, dropout=dropout, concat=concat))
        
        self.convs.append(GATConv(feature_size, num_classes,
                                heads=1, dropout=dropout, concat=False))

    def forward(self, data):
        # NOTE: GATConv does not take edge weights
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x).squeeze()

        for i in range(self.n_layers-1):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[self.n_layers-1](x, edge_index)

        return F.log_softmax(x, dim=1)

    def __str__(self):
        return str(self.n_layers) + "LayerGCN"
    def __repr__(self):
        return str(self.n_layers) + "LayerGCN"
