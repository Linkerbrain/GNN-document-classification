import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class TwoLayerGCNLinearHead(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_labels):
        super(TwoLayerGCNLinearHead, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)

        # feature_size feature_size feature_size feature_size
        self.conv1 = GraphConv(feature_size, feature_size)
        self.conv2 = GraphConv(feature_size, feature_size)

        self.lin1 = torch.nn.Linear(2*feature_size, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_labels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # TODO HANDLE EDGE WEIGHTS

        x = self.embedding(x).squeeze()

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=0)

        return x

    def __str__(self):
        return "TwoLayerGCNLinearHead"