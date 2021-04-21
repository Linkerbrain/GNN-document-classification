import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from tqdm import tqdm

class TransductiveDataset():
    # TODO this is currently a copy of inductive i will do it tommorow bye
    def __init__(self, graph, labels, word_vocab, label_vocab):
        # vocabs for later
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        # debug stats
        self.word_total = 0
        self.word_fails = 0
        self.label_total = 0
        self.label_fails = 0

        # The Data
        self.graphs = self.create_dataobjects(graphs, labels)

        # facts
        self.vocab_size = len(self.word_vocab)
        self.label_count = len(self.label_vocab)

        # print debug stats
        tag = "[dataprep] "
        succes = "Succesfully embedded %d/%d words, %d/%d labels." % (self.word_total-self.word_fails, self.word_total, self.label_total-self.label_fails, self.label_total)
        total_graphs = len(self.graphs)
        ave_length = sum([graph.num_nodes for graph in self.graphs]) / total_graphs
        max_length = max([graph.num_nodes for graph in self.graphs])
        summary = "Dataset contains %d graphs, averaging %d nodes, with a maximum of %d" % (total_graphs, ave_length, max_length)

        print("[dataprep]" + succes + "\n" + tag + summary)

    def get_nodes_idx(self, nodes):
        """returns the corresponding indexes of words, requires ___UNK___ token in vocab"""

        idxs = []
        for node in nodes:
            self.word_total += 1
            if node in self.word_vocab:
                idxs.append(self.word_vocab[node])
            else:
                self.word_fails += 1
                idxs.append(self.word_vocab["___UNK___"])

        return torch.tensor(idxs, dtype=torch.long)

    def get_label_idx(self, label):
        """returns the corresponding index of label, unknown labels get added to first label"""
        self.label_total += 1
        if label in self.label_vocab:
            return torch.tensor([self.label_vocab[label]], dtype=torch.long)
        else:
            self.label_fails += 1
            return torch.tensor([0], dtype=torch.long)


    def create_dataobjects(self, graphs, labels):
        """
        Create PyTorch Geometric Data objects from the raw graphs
        """
        processed_graphs = []

        print("[dataprep] Making Data objects")
        for graph, label in zip(graphs, labels):
            ((edge_indices, edge_weights), nodes) = graph

            nodes_idx = self.get_nodes_idx(nodes)
            label_idx = self.get_label_idx(label)

            torch_edges = torch.tensor(edge_indices, dtype=torch.long)
            torch_edge_weights = torch.tensor(edge_weights, dtype=torch.float)

            graph = Data(x=nodes_idx, edge_index=torch_edges, edge_attr=torch_edge_weights, y=label_idx)
            graph.num_nodes = len(nodes_idx)

            processed_graphs.append(graph)

        return processed_graphs

    def to_dataloader(self, batch_size, shuffle=True, test_size=None):
        """
        Makes a PyTorch Geometric Dataloader that can be used to retrieve batches
        """
        # make a train and test dataloader if test_size is provided
        if test_size:
            random.shuffle(self.graphs)
            split_n = int(len(self.graphs) * (1-test_size))

            return DataLoader(self.graphs[:split_n], batch_size, shuffle), DataLoader(self.graphs[split_n:], batch_size, shuffle)

        return DataLoader(self.graphs, batch_size, shuffle)
    
    def __getitem__(self, index):
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)