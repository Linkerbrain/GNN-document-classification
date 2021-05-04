from dataprep.clean_data import clean_data
from dataprep.balanced_split import get_balanced_split
from dataprep.vocab import vocab

from transductive.graph import transductive_graph
from transductive.dataset import TransductiveDataset
from transductive.trainer import TransductiveTrainer

from models.transductive.gcn import NLayerGCN
from models.transductive.simple_gcn import SimpleGCN
from models.transductive.graph_attention import NLayerGat
from models.transductive.deeper_gcn import NLayerDeeperGCN

# CONFIG

PATH = r"C:\Users\Lodewijk\Desktop\scriptie\GNN-document-classification\data\reuters.train.10000.en"

EPOCHS = 256
MIN_WORD_COUNT = 2 # (for vocab)
FEATURE_SIZE = 128
LEARNING_RATE = 0.01

GRAPH_METHOD = "text gcn paper"

TRY_GPU = True

TRAIN_SIZE = 320
TEST_SIZE = 640

MODEL = {
    "name" : "classic 2 layer GCN",
    "model" : NLayerDeeperGCN,
    "kwargs" : {
        "n_layers" : 4,
        "aggr" : 'softmax',
        "layers_per_layer" : 2,
        "norm_type" : 'layer',
        "dropout" : 0.1
    }
}
# PROGRAM

# - Get Data -
docs, labels = clean_data(PATH)

# get train and test data
train_docs, train_labels, test_docs, test_labels = get_balanced_split(docs, labels, TRAIN_SIZE, TEST_SIZE)

# make vocab
word_vocab, label_vocab = vocab(train_docs+test_docs, labels, min_word_count=MIN_WORD_COUNT)

# create graph
graph = transductive_graph(train_docs+test_docs, GRAPH_METHOD)

# make dataloader
data = TransductiveDataset(graph, train_labels, test_labels, word_vocab, label_vocab)

# make model
model = MODEL["model"](data.vocab_size, FEATURE_SIZE, data.label_count, **MODEL["kwargs"])

# make model trainer
trainer = TransductiveTrainer(data, model, try_gpu=TRY_GPU, lr=LEARNING_RATE)

# train and test!
results = {}

results["loss"] = []
results["accuracy"] = []

for i in range(EPOCHS):
    loss = trainer.train_epoch()
    acc = trainer.validate()
    print("[Epoch %d] Accuracy %.5f Loss %.9f" % (i, acc, loss))

    results["loss"].append(loss)
    results["accuracy"].append(acc)