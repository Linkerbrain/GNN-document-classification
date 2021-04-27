"""
Document Classification using Graph Neural Networks
Using PyTorch Geometric 1.7.0, Cuda 10.2, Pytorch 1.7.0

"""
from dataprep.clean_data import clean_data
from dataprep.vocab import vocab

from transductive.graph import transductive_graph
from transductive.dataset import TransductiveDataset
from transductive.trainer import TransductiveTrainer

def transductive_test(path, graph_method_name, model_class, \
                    epochs, feature_size, min_word_count, try_gpu):
    results = {}

    results["loss"] = []
    results["accuracy"] = []

    # 0 load in data
    docs, labels = clean_data(path)

    # 1 process data
    word_vocab, label_vocab = vocab(docs, labels, min_word_count=min_word_count)

    graph = transductive_graph(docs, graph_method_name)

    # 2 make dataloder
    data = TransductiveDataset(graph, labels, word_vocab, label_vocab)

    # # 3 make & train model
    model = model_class(data.vocab_size, feature_size, data.label_count)

    trainer = TransductiveTrainer(data, model, try_gpu=try_gpu)

    for i in range(epochs):
        loss = trainer.train_epoch()
        acc = trainer.validate()
        print("[Epoch %d] Accuracy %.5f Loss %.5f" % (i, acc, loss))

        results["loss"].append(loss)
        results["accuracy"].append(acc)

    results["model"] = model

    return results
