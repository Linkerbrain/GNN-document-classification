import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import numpy as np

class DataProvider():
    """
    Transforms the parsed graphs into pytorch geometric data objects
    provides a dataloader for training
    """
    def __init__(self, raw_labels, raw_graphs, label_embeddings, word_embeddings):
        # TODO: maybe ask for graph function instead of raw graphs

        # Debug stats
        self.word_total = 0
        self.word_fails = 0
        self.label_total = 0
        self.label_fails = 0

        # create graphs
        self.label_embeddings = label_embeddings
        self.label_size = self.label_embeddings[next(iter(self.label_embeddings))].shape[0]

        self.word_embeddings = word_embeddings
        self.feature_size = self.word_embeddings[next(iter(self.word_embeddings))].shape[0]

        self.graphs = self.embed_graphs(raw_labels, raw_graphs)


    def embed_label(self, label):
        self.label_total += 1
        if label in self.label_embeddings.keys():
            return torch.tensor(self.label_embeddings[label], dtype=torch.long) # maybe float? for if not onehotencoded
        else:
            self.label_fails += 1
            return torch.tensor(np.eye(self.label_size), dtype=torch.long) # we just encode as the first label

    def embed_nodes(self, nodes):
        embedding = []
        for node in nodes:
            self.word_total += 1
            if node in self.word_embeddings.keys():
                embedding.append(self.word_embeddings[node])
            else:
                self.word_fails += 1
                embedding.append(np.random.randn(self.feature_size)) # OOV is random simple vector for now

        return torch.tensor(embedding, dtype=torch.float)

    def embed_graphs(self, raw_labels, raw_graphs):
        all_graphs = []

        for i in range(len(raw_graphs)):
            label = raw_labels[i]
            ((edge_indices, edge_weights), nodes) = raw_graphs[i]

            embedded_label = self.embed_label(label)
            embedded_nodes = self.embed_nodes(nodes)

            torch_edges = torch.tensor(edge_indices, dtype=torch.long)
            torch_edge_weights = torch.tensor(edge_weights, dtype=torch.float)

            graph = Data(x=embedded_nodes, edge_index=torch_edges, edge_attr=torch_edge_weights, y=embedded_label)
            graph.num_nodes = len(embedded_nodes)

            all_graphs.append(graph)

        return all_graphs

    def to_dataloader(self, batch_size, shuffle=True, test_split=1):
        if test_split == 1:
            return DataLoader(self.graphs, batch_size, shuffle)
        
        raise NotImplementedError("We can maybe return multiple dataloaders here for train/dev/test")

    def performance_summary(self):
        tag = "[dataprep] "
        succes = "Succesfully embedded %d/%d words, %d/%d labels." % (self.word_total-self.word_fails, self.word_total, self.label_total-self.label_fails, self.label_total)
        
        total_graphs = len(self.graphs)
        ave_length = sum([graph.num_nodes for graph in self.graphs]) / total_graphs
        max_length = max([graph.num_nodes for graph in self.graphs])
        summary = "Dataset contains %d graphs, averaging %d nodes, with a maximum of %d" % (total_graphs, ave_length, max_length)

        return tag + succes + "\n" + tag + summary
