from numpy.core.numeric import indices
import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import numpy as np

class InductiveTrainer():
    def __init__(self, data, model, val_perc=0.3, batch_size=20, try_gpu=True):
        self.device = "cuda" if try_gpu and torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        print("[gym] Working with device:", self.device)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        self.train_loader, self.test_loader = data.to_dataloader(batch_size, test_size=val_perc)

        self.train_size = len(self.train_loader.dataset)
        self.test_size = len(self.test_loader.dataset)
        self.epoch = 0

        print("[gym] Training on %i, testing on %i" % (self.train_size, self.test_size))

    def train_epoch(self):
        self.model.train()

        loss_all = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            self.optimizer.step()

        self.epoch += 1

        return loss_all / self.train_size

    def validate(self):
        self.model.eval()

        correct = 0
        for data in self.test_loader:
            data = data.to(self.device)
            pred = self.model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / self.test_size

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass