"""
Using PyTorch Geometric

for me it only worked on Cuda 10.2 Pytorch 1.7.0

Pipeline Architecture

"""

from dataprep.clean_data import get_clean_data
from dataprep.corpus_tools import make_mappings
from dataprep.create_graph import graph_unique_coocc
from dataprep.dataprovider import DataProvider

from debug.debug_graph import vis_graph

from gym.trainer import Trainer

from models.simple_gcn import SimpleGCN

PATH = "../data/reuters.train.1000.fr"

LIMIT = -1
EPOCHS = 100
BATCH_SIZE = 60

docs, labels = get_clean_data(PATH)

label_embeddings, word_embeddings = make_mappings(labels, docs, embedding_type="onehot")
raw_graphs = [graph_unique_coocc(d, window_size=2) for d in docs[:LIMIT]]
# label2number, word2embedding = make_mappings(labels, docs, embedding_type="word2vec", w2v_file="mpad/embeddings/GoogleNews-vectors-negative300.bin")

# vis_graph(adjacency, word_idx)
data = DataProvider(labels, raw_graphs, label_embeddings, word_embeddings)
print(data.performance_summary())

dataloader = data.to_dataloader(batch_size=BATCH_SIZE, shuffle=True)

print(next(iter(dataloader)))
print("----")
print(data[0])

model = SimpleGCN(data.feature_size, data.label_size)

# print(model)

trainer = Trainer(dataloader, model, len(raw_graphs))

for i in range(EPOCHS):
    loss = trainer.train_epoch()
    print("[Epoch %d] Loss %.5f" % (i, loss))