import torch
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

class Trainer():
    def __init__(self, dataloader, model=None, datacount = 1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[gym] Working with device:", self.device)

        self.model = model.to(self.device)
        self.dataloader = dataloader


        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

        self.epoch = 0
        self.datacount = datacount

    def train_epoch(self):
        if not self.model:
            print("[gym] No model loaded!")
            return

        self.model.train()

        loss_all = 0
        for data in tqdm(self.dataloader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            self.optimizer.step()
        return loss_all / self.datacount

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass