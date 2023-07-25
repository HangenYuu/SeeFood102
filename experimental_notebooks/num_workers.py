import timm
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torch.nn import functional as F
import time
from collections import OrderedDict, namedtuple
from itertools import product
from time import time
import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('swin_large_patch4_window7_224', pretrained=False, num_classes=101).to(device)
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)

train_dataset = Food101(root='./data', split='train', transform=transform)
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time()

        self.run_params = run
        self.run_count += 1

        self.network = network
        self.loader = loader

    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time() - self.epoch_start_time
        run_duration = time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

    def track_loss(self, loss):
        self.epoch_loss += loss.item() * self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, filename):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{filename}.csv')

params = OrderedDict([
    ("lr", [0.01]),
    ("batch_size", [16, 32, 64]), 
    ("num_workers", [0, 1, 2, 4, 8])
])

m = RunManager()

for run in RunBuilder.get_runs(params):
    loader = DataLoader(train_dataset, batch_size=run.batch_size, num_workers=run.num_workers)
    optimizer = optim.Adam(model.parameters(), lr=run.lr)

    m.begin_run(run, model, loader)

    for epoch in range(1):
        m.begin_epoch()
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
        m.end_epoch()
    m.end_run()
m.save('results')
