import torch
from torch import optim
from tqdm import tqdm

class Trainer():
    def __init__(self, dataloader, model=None,
                lr=0.1):
        self.model = model
        self.dataloader = dataloader

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[gym] Working with device:", self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.epoch = 0

    def train_epoch(self):
        if not self.model:
            print("[gym] No model loaded!")
            return

        for batch_ix, batch in enumerate(self.dataloader):
            batch = (t.to(self.device) for t in batch)
            A, nodes, y, n_graphs = batch

            preds = self.model(nodes, A)

            print(preds)

            loss = self.criterion(preds, y)

            self.optimizer.zero_grad()
            loss.backward()

            # grad norm clipping?
            self.optimizer.step()
            self.scheduler.step()

            print("Batch %d, loss %.5f" % (batch_ix, loss.item()))

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass