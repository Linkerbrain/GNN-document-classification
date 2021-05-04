import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from tqdm import tqdm

class TransductiveDataset():
    def __init__(self, graph, train_labels, test_labels, word_vocab, label_vocab):
        # vocabs for later
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        # debug stats
        self.word_total = 0
        self.word_fails = 0
        self.label_total = 0
        self.label_fails = 0

        # The Data (the big graph, the labels and what nodes the labels correspond to)
        self.graph, self.train_labels, self.train_idx, self.test_labels, self.test_idx = self.create_dataobject(graph, train_labels, test_labels)

        # facts
        self.vocab_size = len(self.word_vocab)
        self.label_count = len(self.label_vocab)

        # print debug stats
        tag = "[dataprep] "
        succes = "Succesfully embedded %d/%d words, %d/%d labels." % (self.word_total-self.word_fails, self.word_total, self.label_total-self.label_fails, self.label_total)
        graph_length = self.graph.num_nodes
        summary = "Dataset contains %d graph with %d nodes" % (1, graph_length)

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

    def create_dataobject(self, graph, train_labels, test_labels):
        """
        Create PyTorch Geometric Data objects from the raw graph and save labels and the nodes they correspond to
        """

        print("[dataprep] Making Data object")
        ((edge_indices, edge_weights), nodes) = graph

        nodes = self.get_nodes_idx(nodes)
        train_label_tensor = self.get_labels_idx(train_labels)
        test_label_tensor = self.get_labels_idx(test_labels)

        torch_edges = torch.tensor(edge_indices, dtype=torch.long)
        torch_edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        graph = Data(x=nodes, edge_index=torch_edges, edge_attr=torch_edge_weights)
        graph.num_nodes = len(nodes)

        # WE ASSUME THE TRAIN, TEST DOCUMENTS ARE THE FIRST X AND SECOND Y NODES IN GRAPH RESPECTIVELY
        train_amount = len(train_labels)
        test_amount = len(test_labels)
        train_idx = torch.tensor(list(range(train_amount)), dtype=torch.long)
        test_idx = torch.tensor(list(range(train_amount, train_amount+test_amount)), dtype=torch.long)

        print("Train shape:", train_idx.shape)
        print("TEst shape:", test_idx.shape)

        return graph, train_label_tensor, train_idx, test_label_tensor, test_idx

    def get_train_test_data(self):
        """
        Returns train and test labels, with the indexes they correspond to
        """
        return self.train_labels, self.train_idx, \
                self.test_labels, self.test_idx
    
    def __getitem__(self, index):
        return self.labels[index], self.node_idx_of_labels[index]

    def __len__(self):
        return len(self.labels)