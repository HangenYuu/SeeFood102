import lightning as L
import torch
import timm
import torch.nn.functional as F
import torchmetrics
from typing import Union
from pathlib import Path
from torchvision.datasets import Food101
from torch.utils.data import random_split, DataLoader

class Food101DataModule(L.LightningDataModule):
    def __init__(self, train_transform, test_transform, data_dir: Union[str, Path] = "data", batch_size: int = 128) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        Food101(self.data_dir, split='train', download=True) # type: ignore
        Food101(self.data_dir, split='test', download=True) # type: ignore

    def setup(self, stage: str = 'fit'):
        if stage == 'fit':
            food101_train = Food101(self.data_dir, split='train', download=True, transform=self.train_transform) # type: ignore
            food101_test = Food101(self.data_dir, split='train', download=True, transform=self.test_transform) # type: ignore
            self.food101_train, _ = random_split(food101_train, [0.8, 0.2], generator=torch.Generator().manual_seed(42)) # type: ignore
            _, self.food101_val = random_split(food101_test, [0.8, 0.2], generator=torch.Generator().manual_seed(42)) # type: ignore

        if stage == 'test':
            self.food101_test = Food101(self.data_dir, split='test', download=True, transform=self.test_transform) # type: ignore

        if stage == "predict":
            self.food101_predict = Food101(self.data_dir, split='test', download=True, transform=self.test_transform) # type: ignore

    def train_dataloader(self):
        return DataLoader(self.food101_train, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.food101_val, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.food101_test, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.food101_predict, batch_size=self.batch_size, num_workers=4, pin_memory=True)

class Food101Classifier(L.LightningModule):
    def __init__(self, model_name: str = "hf_hub:timm/levit_256.fb_dist_in1k") -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = 101
        self.model = timm.create_model(model_name, pretrained=True, num_classes=101)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=101)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=101)
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=101)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, 1)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_acc(preds, labels)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        self.model.eval()
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, 1)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.valid_acc(preds, labels)
        self.log('val_acc', self.valid_acc, prog_bar=True, sync_dist=True)
        self.f1_metric(preds, labels)
        self.log("val_f1", self.f1_metric, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        self.model.eval()
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, 1)
        loss = F.cross_entropy(outputs, labels)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        self.valid_acc(preds, labels)
        self.log('test_acc', self.valid_acc, prog_bar=True, sync_dist=True)
        self.f1_metric(preds, labels)
        self.log("test_f1", self.f1_metric, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001, foreach=True)