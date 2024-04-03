"""MNIST autoencoder example.

To run: python autoencoder.py --trainer.max_epochs=50

"""

from os import path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from lightning.pytorch import (
    LightningDataModule,
    LightningModule,
    Trainer,
    callbacks,
    cli_lightning_logo,
)
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.demos.mnist_datamodule import MNIST
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE
from torch import nn
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder, LearningRateMonitor, RichModelSummary, RichProgressBar, ThroughputMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image


from dataset import ECG_PPG_Dataset
import torchmetrics


class ECGPPG(LightningModule):
    def __init__(
        self, input_dim=2216, output_dim=104, lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()

        self.train_r2 = torchmetrics.R2Score(output_dim)
        self.val_r2 = torchmetrics.R2Score(output_dim)
        self.test_r2 = torchmetrics.R2Score(output_dim)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(0.5),
            nn.Conv1d(16, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # nn.Dropout(0.5),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(
            nn.Linear(2112, 1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        # self.net = nn.Sequential(
        #     nn.Conv1d(1, 8, kernel_size=16, stride=2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2),
        #     nn.Flatten(),
        #     nn.Dropout(0.5),
        #     nn.Linear(4400, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, output_dim),
        # )

    def forward(self, x):
        x = self.conv(x)
        # x = torch.squeeze(x)
        y_pred = self.linear(x)
        # y_pred = self.net(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _common_step(self, batch, batch_idx, stage: str):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        r2 = torchmetrics.functional.r2_score(y_pred, y)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/r2", r2, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class DataModule(LightningDataModule):
    def __init__(self, data_dir="/work/app/wanghongyang/dataset/ecg_ppg_feat_map", batch_size: int = 512, num_workers: int = 4):
        super().__init__()
        dataset = ECG_PPG_Dataset(data_dir)
        
        self.test_set = None
        self.train_set, self.val_set = random_split(
            dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size)

class Project:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

class CustomCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(
            Project, "project", instantiate=False, sub_configs=True
        )
        

def cli_main():
    cli = CustomCLI(
        ECGPPG,
        DataModule,
        seed_everything_default=1234,
        run=True,  # used to de-activate automatic fitting.
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    # cli.trainer.fit(
    #     cli.model,
    #     datamodule=cli.datamodule,
    # )
    # cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
    # predictions = cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)
    # print(predictions[0])


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
