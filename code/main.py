"""
Using PyTorch Geometric

for me it only worked on Cuda 10.2 Pytorch 1.7.0

Pipeline Architecture

0. Raw (Reuters) Data (big .tsv file)
-   clean_data.py [path -> labels, docs]:
|       load_file 
|       make_ascii
|       clean_str
V
1. Clean Data (labels (list of strings), docs (list of lists of strings)
-   corpus_tools.py [labels, docs -> label2number, word2embedding]
|       make label dic
|       make corpus vocab
|       embed corpus coab
|  - create_graph.py [docs -> list of raw_graphs (edge_indexes, words)]
| |     create doc graph
--- dataloader.py [labels, raw_graphs, label2number, word2embedding -> DataLoader]
|       embed labels
|       embed words
|       combine graphs
V
2. Processed Data (PyTorch DataLoader providing num_label, A, embeddings)
|
|
|
|
V
3. trained model
|
|
|
V
4. Evaluation

"""

from dataprep.clean_data import get_clean_data
from dataprep.corpus_tools import make_mappings
from dataprep.create_graph import graph_unique_coocc
from dataprep.dataprovider import DataProvider

from debug.debug_graph import vis_graph

from gym.trainer import Trainer

from models.gcn import GCN

PATH = "GNN-document-classification/data/reuters.train.1000.fr"

HIDDEN_LAYER_SIZE = 100
DROPOUT = 0.01

docs, labels = get_clean_data(PATH)

label_embeddings, word_embeddings = make_mappings(labels, docs, embedding_type="onehot")
raw_graphs = [graph_unique_coocc(d, window_size=2) for d in docs]
# label2number, word2embedding = make_mappings(labels, docs, embedding_type="word2vec", w2v_file="mpad/embeddings/GoogleNews-vectors-negative300.bin")

# vis_graph(adjacency, word_idx)
data = DataProvider(labels, raw_graphs, label_embeddings, word_embeddings)
print(data.performance_summary())

dataloader = data.to_dataloader(batch_size=3, shuffle=True)

print(next(iter(dataloader)))

# model = GCN(data.feature_count, HIDDEN_LAYER_SIZE, data.class_count, DROPOUT)

# print(model)

# trainer = Trainer(dataloader, model)

# trainer.train_epoch()