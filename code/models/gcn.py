import torch.nn as nn
import torch.nn.functional as F
from .graph_conv import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        x = F.log_softmax(x, dim=1)
        return x
