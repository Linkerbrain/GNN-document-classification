import scipy.sparse as sp
from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import torch
from torch.utils.data import (
	Dataset,
	DataLoader,
)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)

class DocumentGraphDataset(Dataset):
	def __init__(self, raw_labels, raw_graphs, label2number, word2embedding):
		"""
		Dataset inheriting from PyTorch's Dataset class.
		Constructs a graph of words of each document based
		"""
		self.label2number = label2number
		self.word2embedding = word2embedding

		self.feature_count = self.word2embedding[next(iter(self.word2embedding))].shape[0]
		self.class_count = len(self.label2number.keys())

		self.graphs = self.embed_graphs(raw_graphs)
		self.labels = self.embed_labels(raw_labels)

	def embed_graphs(self, raw_graphs):
		# This can also be done in the model, that's prob better TODO

		As, docs = zip(*raw_graphs)

		embeddings = []

		total_count = 0
		fail_count = 0
		for doc in docs:
			doc_embed = []
			for word in doc:
				total_count += 1
				if word in self.word2embedding.keys():
					doc_embed.append(self.word2embedding[word])
				else:
					fail_count += 1
					doc_embed.append(np.random.randn(self.feature_count)) # TODO: handle out of vocab better?
			embeddings.append(np.array(doc_embed))
		
		print("[dataprep] Found embedding of %d/%d words" % (total_count-fail_count, total_count))
		return list(zip(As, embeddings))

	def embed_labels(self, raw_labels):
		labels = [self.label2number[l] for l in raw_labels]

		return labels

	def __len__(self):
		return len(self.graphs)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		A, nodes = self.graphs[idx]
		y = self.labels[idx]

		return (A, nodes, y)

	def prepare_batch(self, batch):
		batch_A, batch_nodes, batch_y = zip(*batch)

		n_graphs = len(batch_nodes)

		max_n_nodes = max([nodes.shape[0] for nodes in batch_nodes])
		embedding_size = batch_nodes[0].shape[1]
		n_nodes = n_graphs * max_n_nodes

		combined_A = lil_matrix((n_nodes, n_nodes))
		combined_features = np.zeros((n_nodes, embedding_size))
		for i, (A, features) in enumerate(zip(batch_A, batch_nodes)):
			start_ix = i * max_n_nodes
			combined_A[start_ix: start_ix + A.shape[0], start_ix: start_ix + A.shape[0]] = A
			combined_features[start_ix: start_ix + features.shape[0]] = features

		# convert to PyTorch data
		torch_A = sparse_mx_to_torch_sparse_tensor(combined_A)
		torch_features = torch.FloatTensor(combined_features)
		torch_y = torch.FloatTensor(batch_y)

		return torch_A, torch_features, torch_y, torch.FloatTensor([n_graphs])

	def to_dataloader(self, batch_size, shuffle, drop_last):
		return DataLoader(
			self,
			batch_size=batch_size,
			shuffle=shuffle,
			drop_last=drop_last,
			collate_fn=self.prepare_batch,
		)
