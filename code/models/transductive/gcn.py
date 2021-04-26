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
        x, edge_index = data.x, data.edge_index

        x = self.embedding(x).squeeze()

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def __str__(self):
        return "TwoLayerGCN"
