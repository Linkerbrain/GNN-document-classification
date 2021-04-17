"""
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
|  - create_graph.py [docs -> list of raw_graphs (Adjacency, index2word)]
| |     create doc graph
--- data_loader.py [labels, raw_graphs, label2number, word2embedding -> DataLoader]
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


from dataprep.dataloader import DocumentGraphDataset

from debug.debug_graph import vis_graph

PATH = "GNN-document-classification/data/reuters.train.1000.fr"

docs, labels = get_clean_data(PATH)
print("First doc (%s)\n" % labels[0], docs[0])

adjacency, words = graph_unique_coocc(docs[0], window_size=2)

label2number, word2embedding = make_mappings(labels, docs, embedding_type="onehot")

# vis_graph(adjacency, word_idx)

data = DocumentGraphDataset(labels, (adjacency, words), label2number, word2embedding)