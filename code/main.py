import pandas as pd
import os

from inductive.test_model import inductive_test
from transductive.test_model import transductive_test

from models.inductive.gcn import TwoLayerGCNLinearHead
from models.transductive.gcn import TwoLayerGCN

inductive_model = TwoLayerGCNLinearHead
transductive_model = TwoLayerGCN

# config
DATA_PATH = r"C:\Users\Lodewijk\Desktop\scriptie\GNN-document-classification\data\reuters.train.10000.fr"

EPOCHS = 128
BATCH_SIZE = 256

FEATURE_SIZE = 128

MIN_WORD_COUNT = 2 # (for vocab)

# inductive_test(DATA_PATH, "unique co-occurence", inductive_model, epochs=EPOCHS, feature_size=FEATURE_SIZE, min_word_count=MIN_WORD_COUNT, batch_size=BATCH_SIZE)
transductive_test(DATA_PATH, "text gcn paper", transductive_model, epochs=EPOCHS, feature_size=FEATURE_SIZE, min_word_count=MIN_WORD_COUNT, batch_size=BATCH_SIZE)