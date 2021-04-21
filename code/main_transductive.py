"""
Document Classification using Graph Neural Networks
Using PyTorch Geometric 1.7.0, Cuda 10.2, Pytorch 1.7.0

Pipeline Architecture

0 ---------------------------- some doc dataset ---------------------------------------
                                    |
                               clean_data.py
                                    |
                                    V
1 ---------------------------- docs, labels -------------------------------
                   |                                  |
inductive_graph.py or transductive_graph.py        vocab.py 
                   |                                  |
                   V                                  V
2 -------  edge_indices, nodes --------  word_vocab, label_vocab ----------
                                    |
                                    |
            inductive_dataloader.py or transductive_dataloader.py
                                    |
                                    |
                                    V
3 ----------------------------  dataloader  -------------------------------
                                    |
                                    |
                                trainer.py   <========== Model
                                    |
                                    |
                                    V
4 -#-#-#-#-#-#-#-#-#-#-#-#-#   Results   #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

"""
from dataprep.clean_data import clean_data
from dataprep.vocab import vocab
from dataprep.transductive_graph import transductive_graph

from dataprep.transductive_dataset import TransductiveDataSet

from gym.trainer import Trainer

from models.simple_gcn import SimpleGCN


# Hyperparameters
PATH = "../data/reuters.train.1000.fr"
MIN_WORD_COUNT = 2
FEATURE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 100

# Da program
# 0 load in data
docs, labels = clean_data(PATH)

# 1 process data
word_vocab, label_vocab = vocab(docs, labels, min_word_count=MIN_WORD_COUNT)

graph = transductive_graph(docs)

# 2 make dataloder
# data = TransductiveDataSet(graph, labels, word_vocab, label_vocab)

# print(data[0], next(iter(data.to_dataloader(60))))

# # 3 make & train model
# model = SimpleGCN(data.vocab_size, FEATURE_SIZE, data.label_count)

# trainer = Trainer(data, model, batch_size=BATCH_SIZE)

# for i in range(EPOCHS):
#     loss = trainer.train_epoch()
#     acc = trainer.validate()
#     print("[Epoch %d] Accuracy %.5f Loss %.5f" % (i, acc, loss))