import pandas as pd
import os
import re

# this is not very pretty but enough to test all the models on the infrastructure 

from inductive.test_model import inductive_test
from transductive.test_model import transductive_test

from models.inductive.gcn import TwoLayerGCNLinearHead
from models.inductive.topkpooling import TopK
from models.inductive.copypasted import ModelProteinBest

from models.transductive.gcn import TwoLayerGCN
from models.transductive.copypasted import GAT, SuperGAT, AGNN

inductive_models = [TwoLayerGCNLinearHead, TopK, ModelProteinBest]
transductive_models = [TwoLayerGCN, GAT, SuperGAT, AGNN]

# config
TEST_NAME = "firstnight"

DATA_PATH = r"C:\Users\Lodewijk\Desktop\scriptie\GNN-document-classification\data\reuters.train.10000.fr"
RESULT_FOLDER = r"C:\Users\Lodewijk\Desktop\scriptie\GNN-document-classification\results\ "

EPOCHS = 160
BATCH_SIZE = 256

FEATURE_SIZE = 128

MIN_WORD_COUNT = 2 # (for vocab)

# prepare
result_location = RESULT_FOLDER + TEST_NAME + '\\'
os.makedirs(result_location, exist_ok=True)

# transductive
for model in transductive_models:
    print(" --- Now Testing: --- ", str(model))
    results = transductive_test(DATA_PATH, "text gcn paper", model, epochs=EPOCHS, feature_size=FEATURE_SIZE, min_word_count=MIN_WORD_COUNT, batch_size=BATCH_SIZE)

    result_df = pd.DataFrame(results)

    name = str(model)
    name = re.sub(r"[^A-Za-z0-9.]", "", name)
    result_df.to_csv(result_location + name + ".csv")

# inductive
for model in inductive_models:
    print(" --- Now Testing: --- ", str(model))
    results = inductive_test(DATA_PATH, "unique co-occurence", model, epochs=EPOCHS, feature_size=FEATURE_SIZE, min_word_count=MIN_WORD_COUNT, batch_size=BATCH_SIZE)

    result_df = pd.DataFrame(results)

    name = str(model)
    name = re.sub(r"[^A-Za-z0-9.]", "", name)
    result_df.to_csv(result_location + name + ".csv")
