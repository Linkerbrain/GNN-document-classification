import pandas as pd
import os

from inductive.validate_model import inductive_test
from transductive.validate_model import transductive_test

from models.inductive.gcn import TwoLayerGCNLinearHead
from models.transductive.gcn import TwoLayerGCN

# config
DATA_PATH = r"C:\Users\Lodewijk\Desktop\scriptie\GNN-document-classification\data\reuters.train.1000.fr"

EPOCHS = 128
BATCH_SIZE = 256

FEATURE_SIZE = 128

MIN_WORD_COUNT = 2 # (for vocab)

TRY_GPU = True

inductive_test(DATA_PATH, "unique co-occurence", TwoLayerGCNLinearHead, epochs=EPOCHS, feature_size=FEATURE_SIZE, min_word_count=MIN_WORD_COUNT, batch_size=BATCH_SIZE, try_gpu=TRY_GPU)
# transductive_test(DATA_PATH, "text gcn paper", TwoLayerGCN, epochs=EPOCHS, feature_size=FEATURE_SIZE, min_word_count=MIN_WORD_COUNT, batch_size=BATCH_SIZE, try_gpu=TRY_GPU)