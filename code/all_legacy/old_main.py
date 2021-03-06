"""
Document Classification using Graph Neural Networks
Using PyTorch Geometric 1.7.0, Cuda 10.2, Pytorch 1.7.0

Pipeline Architecture

0 ---------------------------- some doc dataset ---------------------------------------
                                    |
                               clean_data.py
                                    |
                                    V
1 ---------------------------- docs, labels -------------------------------------------
                   |                                  |
inductive_graph.py or transductive_graph.py        vocab.py 
                   |                                  |
                   V                                  V
2 -------  edge_indices, nodes --------  word_vocab, doc_vocab, label_vocab ----------
                                    |
                                    |
            inductive_dataloader.py or transductive_dataloader.py
                                    |
                                    |
                                    V
3 ----------------------------  dataloader  -------------------------------------------
                                    |
                                    |
                                trainer.py   <========== Model
                                    |
                                    |
                                    V
4 -#-#-#-#-#-#-#-#-#-#-#-#-#   Results   #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

"""

from dataprep.clean_data import get_clean_data
from dataprep.corpus_tools import make_mappings
from dataprep.create_graph import graph_unique_coocc
from dataprep.dataprovider import DataProvider

from debug.debug_graph import vis_graph

from gym.trainer import Trainer

from models.simple_gcn import SimpleGCN

PATH = "../data/reuters.train.10000.fr"

LIMIT = None
EPOCHS = 100
BATCH_SIZE = 100

docs, labels = get_clean_data(PATH)

# label_embeddings, word_embeddings = make_mappings(labels, docs, embedding_type="word2vec", w2v_file=r"C:\Users\Lodewijk\Desktop\scriptie\mpad\embeddings\GoogleNews-vectors-negative300.bin")
label_embeddings, word_embeddings = make_mappings(labels, docs, embedding_type="index")

raw_graphs = [graph_unique_coocc(d, window_size=2) for d in docs[:LIMIT]]

# vis_graph(adjacency, word_idx)
data = DataProvider(labels, raw_graphs, label_embeddings, word_embeddings)
print(data.performance_summary())

print(data[0], next(iter(data.to_dataloader(60))))

model = SimpleGCN(len(word_embeddings), 300, 4)

# print(model)

trainer = Trainer(data, model)

for i in range(EPOCHS):
    loss = trainer.train_epoch()
    acc = trainer.validate()
    print("[Epoch %d] Accuracy %.5f Loss %.5f" % (i, acc, loss))