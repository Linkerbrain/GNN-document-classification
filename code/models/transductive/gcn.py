import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TwoLayerGCN(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes):
        super(TwoLayerGCN, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)
        
        self.conv1 = GCNConv(feature_size, feature_size)
        self.conv2 = GCNConv(feature_size, num_classes)

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x).squeeze()

        x = self.conv1(x, edge_index, edge_weight=edge_weights)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weights)

        return F.log_softmax(x, dim=1)

    def __str__(self):
        return "TwoLayerGCN"

class NLayerGCN(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_classes,
                 n_layers, dropout):
        super(NLayerGCN, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)
        
        self.convs = torch.nn.ModuleList()

        for i in range(n_layers-1):
            self.convs.append(GCNConv(feature_size, feature_size))
        self.convs.append(GCNConv(feature_size, num_classes))

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr

        x = self.embedding(x).squeeze()

        for i in range(self.n_layers-1):
            x = self.convs[i](x, edge_index, edge_weight=edge_weights)
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training)

        x = self.convs[self.n_layers-1](x, edge_index, edge_weight=edge_weights)

        return F.log_softmax(x, dim=1)

    def __str__(self):
        return str(self.n_layers) + "LayerGCN"
    def __repr__(self):
        return str(self.n_layers) + "LayerGCN"