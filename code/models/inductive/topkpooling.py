import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class TopK(torch.nn.Module):
    def __init__(self, vocab_size, feature_size, num_labels):
        super(TopK, self).__init__()

        self.embedding = torch.nn.Embedding(vocab_size, feature_size)

        # feature_size feature_size feature_size feature_size
        self.conv1 = GraphConv(feature_size, feature_size)
        self.pool1 = TopKPooling(feature_size, ratio=0.8)
        self.conv2 = GraphConv(feature_size, feature_size)
        self.pool2 = TopKPooling(feature_size, ratio=0.8)
        self.conv3 = GraphConv(feature_size, feature_size)
        self.pool3 = TopKPooling(feature_size, ratio=0.8)

        self.lin1 = torch.nn.Linear(feature_size*2, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_labels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # TODO HANDLE EDGE WEIGHTS

        x = self.embedding(x).squeeze()

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=0)

        return x

    def __str__(self):
        return "TopK"