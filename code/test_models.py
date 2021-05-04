import pandas as pd
import os
from datetime import datetime

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

EXPERIMENT_NAME = "quick_test"

PATH = r"C:\Users\Lodewijk\Desktop\scriptie\GNN-document-classification\data\reuters.train.10000.en"

EPOCHS = 150
MIN_WORD_COUNT = 2 # (for vocab)
FEATURE_SIZE = 128
LEARNING_RATE = 0.01

GRAPH_METHOD = "text gcn paper"

TRY_GPU = True

TRAIN_SIZES = [40, 80, 160, 320, 640, 1280]
TEST_SIZES = [80, 160, 320, 640, 2560]

MODELS = [
    {
        "name" : "Vanilla 1 Layer GCN",
        "model" : NLayerGCN,
        "kwargs" : {
            "n_layers" : 1,
            "dropout" : 0.1
        }
    },
    {
        "name" : "Vanilla 2 Layer GCN",
        "model" : NLayerGCN,
        "kwargs" : {
            "n_layers" : 2,
            "dropout" : 0.1
        }
    },
    {
        "name" : "Vanilla 4 Layer GCN",
        "model" : NLayerGCN,
        "kwargs" : {
            "n_layers" : 4,
            "dropout" : 0.1
        }
    },
    {
        "name" : "1 hop SimpleGCN",
        "model" : SimpleGCN,
        "kwargs" : {
            "hops" : 1,
        }
    },
    {
        "name" : "3 hop SimpleGCN",
        "model" : SimpleGCN,
        "kwargs" : {
            "hops" : 3,
        }
    },
    {
        "name" : "2 Layer 4 Heads Concat GatGCN",
        "model" : NLayerGat,
        "kwargs" : {
            "n_layers" : 2,
            "n_heads" : 4,
            "concat" : True,
            "dropout" : 0.1
        }
    },
    {
        "name" : "2 Layer 2 Heads Concat GatGCN",
        "model" : NLayerGat,
        "kwargs" : {
            "n_layers" : 2,
            "n_heads" : 2,
            "concat" : True,
            "dropout" : 0.1
        }
    },
    {
        "name" : "2 Layer 4 Heads NonConcat GatGCN",
        "model" : NLayerGat,
        "kwargs" : {
            "n_layers" : 2,
            "n_heads" : 4,
            "concat" : False,
            "dropout" : 0.1
        }
    },
]

# PROGRAM

# prepare output
experiment_folder = r"C:\Users\Lodewijk\Desktop\scriptie\GNN-document-classification\results/"+EXPERIMENT_NAME+"/"
if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

all_results_file = experiment_folder + "results.csv"
if not os.path.exists(experiment_folder):
    dummy_df = pd.DataFrame({"model_name":[],"train_size":[],"test_size":[],"final_acc":[]})
    dummy_df.to_csv(all_results_file, header=True, index=False)

# - Get Data -
docs, labels = clean_data(PATH)

# - Iterate over data sizes
for model_config in MODELS:
    for train_size in TRAIN_SIZES:
        for test_size in TEST_SIZES:
                print("\n Now Working on [(%d, %d), %s]\n" % (train_size, test_size, model_config["name"]))

                # get train and test data
                train_docs, train_labels, test_docs, test_labels = get_balanced_split(docs, labels, train_size, test_size)

                # make vocab
                word_vocab, label_vocab = vocab(train_docs+test_docs, labels, min_word_count=MIN_WORD_COUNT)

                # create graph
                graph = transductive_graph(train_docs+test_docs, GRAPH_METHOD)

                # make dataloader
                data = TransductiveDataset(graph, train_labels, test_labels, word_vocab, label_vocab)

                # make model
                model = model_config["model"](data.vocab_size, FEATURE_SIZE, data.label_count, **model_config["kwargs"])

                # make model trainer
                trainer = TransductiveTrainer(data, model, try_gpu=TRY_GPU, lr=LEARNING_RATE)

                # train and test!
                progression = {}

                progression["loss"] = []
                progression["accuracy"] = []

                failed=False
                for i in range(EPOCHS):
                    try:
                        loss = trainer.train_epoch()
                        acc = trainer.validate()
                        print("[Epoch %d] Accuracy %.5f Loss %.9f" % (i, acc, loss))

                        progression["loss"].append(loss)
                        progression["accuracy"].append(acc)
                    except:
                        test_name = "%s on %d,%d" % (model_config["name"], train_size, test_size)
                        print("Could not perform test %s, probably ran out of Cuda Memory")
                        failed = True
                        break

                if failed:
                    continue

                progression_df = pd.DataFrame(progression)
                
                now = datetime.now()
                dt_string = now.strftime("%d %H %M %S")   
                test_name = "%s-%d-%d-%s.csv" % (model_config["name"], train_size, test_size, dt_string)
                progression_df.to_csv(experiment_folder+test_name)

                summary_df = pd.DataFrame({"model_name":[model_config["name"]],"train_size":[train_size],"test_size":[test_size],"final_acc":[acc]})
                summary_df.to_csv(all_results_file, mode='a', header=False, index=False)