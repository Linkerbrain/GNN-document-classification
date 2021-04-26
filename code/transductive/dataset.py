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

        # The Data (the big graph, the labels and what nodes the labels correspond to)
        self.graph, self.labels, self.node_idx_of_labels = self.create_dataobject(graph, labels)

        # facts
        self.vocab_size = len(self.word_vocab)
        self.label_count = len(self.label_vocab)

        # print debug stats
        tag = "[dataprep] "
        succes = "Succesfully embedded %d/%d words, %d/%d labels." % (self.word_total-self.word_fails, self.word_total, self.label_total-self.label_fails, self.label_total)
        graph_length = self.graph.num_nodes
        summary = "Dataset contains %d graphs  with %d nodes" % (1, graph_length)

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

    def get_labels_idx(self, labels):
        """returns the corresponding index of label, unknown labels get added to first label"""
        label_idxs = []
        for label in labels:
            self.label_total += 1
            if label in self.label_vocab:
                label_idxs.append(self.label_vocab[label])
            else:
                self.label_fails += 1
                label_idxs.append(0)

        return torch.tensor(label_idxs, dtype=torch.long)

    def create_dataobject(self, graph, labels):
        """
        Create PyTorch Geometric Data objects from the raw graph and save labels and the nodes they correspond to
        """

        print("[dataprep] Making Data object")
        ((edge_indices, edge_weights), nodes) = graph

        nodes = self.get_nodes_idx(nodes)
        labels = self.get_labels_idx(labels)

        torch_edges = torch.tensor(edge_indices, dtype=torch.long)
        torch_edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        graph = Data(x=nodes, edge_index=torch_edges, edge_attr=torch_edge_weights)
        graph.num_nodes = len(nodes)

        # WE ASSUME THE DOCUMENTS ARE THE FIRST N NODES IN GRAPH
        node_idx_of_labels = torch.tensor(list(range(len(labels))), dtype=torch.long)

        return graph, labels, node_idx_of_labels

    def get_train_test_data(self, test_size):
        """
        Returns train and test labels, with the indexes they correspond to
        """
        shuffled_idx = torch.randperm(len(self.labels))

        train_amount = int(len(shuffled_idx) * (1-test_size))
        train_idx = shuffled_idx[:train_amount]
        test_idx = shuffled_idx[train_amount:]

        return self.labels[train_idx], self.node_idx_of_labels[train_idx], \
                self.labels[test_idx], self.node_idx_of_labels[test_idx]
    
    def __getitem__(self, index):
        return self.labels[index], self.node_idx_of_labels[index]

    def __len__(self):
        return len(self.labels)