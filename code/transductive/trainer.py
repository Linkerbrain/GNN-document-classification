from numpy.core.numeric import indices
import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import numpy as np

class TransductiveTrainer():
    def __init__(self, data, model, val_perc=0.3):
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        print("[gym] Working with device:", self.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.02) # note big lr

        self.graph = data.graph
        self.train_labels, self.train_label_idxs, \
            self.test_labels, self.test_label_idxs = data.get_train_test_data(test_size=val_perc)

        self.train_size = len(self.train_labels)
        self.test_size = len(self.test_labels)
        self.epoch = 0

        print("[gym] Training on %i, testing on %i" % (self.train_size, self.test_size))

    def train_epoch(self):
        self.model.train()

        self.optimizer.zero_grad()
        preds = self.model(self.graph)

        train_preds = preds[self.train_label_idxs]

        loss = F.nll_loss(train_preds, self.train_labels)
        loss.backward()

        self.optimizer.step()

        self.epoch += 1

        return loss / self.train_size

    def validate(self):
        # This could be done in the train step as well since it does not require much extra calculation

        self.model.eval()

        preds = self.model(self.graph)

        test_preds = preds[self.test_label_idxs].max(dim=1).indices

        correct = test_preds.eq(self.test_labels).sum().item()

        return correct / self.test_size

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass