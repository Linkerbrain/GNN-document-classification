from clean_data import get_clean_data
from create_graph import graph_unique_coocc, add_master_node
from debug_graph import vis_graph

PATH = "GNN-document-classification/data/reuters.train.1000.fr"

docs, labels = get_clean_data(PATH)
print("First doc (%d)\n" % labels[0], docs[0])

adjacency, word_idx = graph_unique_coocc(docs[0], window_size=2)

# adjacency, word_idx = add_master_node(adjacency, word_idx)

vis_graph(adjacency, word_idx)