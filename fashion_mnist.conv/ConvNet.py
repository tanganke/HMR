import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import torchmetrics
import pytorch_lightning as pl

from pyutils.base import log


class ConvNet(pl.LightningModule):
    def __init__(self, *, in_channels: int, output_classes: int,
                 classes: list = None, num_samples: int = None, learning_rate=1e-4):
        R"""
        3-layer convolutional network.
        input: `in_channels`*28*28 image, for instance MNIST, Fashion-MNIST dataset.

        see also: https://github.com/tensorflow/tpu/blob/master/tools/colab/fashion_mnist.ipynb

        Args:
            in_channels (int)
            output_classes (int)
        """

        super().__init__()
        self.save_hyperparameters()
        log.info(f'Constructing `ConvNet`, hparams =\n{self.hparams}.')
        self.output_classes = output_classes
        self.classes = classes  # NOTE necessary
        self.num_samples = num_samples  # NOTE necessary

        self.feature = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 64, (5, 5), padding=2),
            nn.ELU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(0.25),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (5, 5), padding=2),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (5, 5), padding=2),
            nn.ELU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.logits = nn.Sequential(
            nn.Linear(256, output_classes)
        )

        self.train_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        """
        Returns:
            Tensor: logits
        """
        # in lightning, forward defines the prediction/inference actions
        feature = self.feature(x)
        logits = self.logits(feature)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        # compute accuracy
        pred = logits.softmax(-1)
        self.train_acc(pred, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        self.log('train_acc', self.train_acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = logits.softmax(-1)
        self.test_acc(pred, y)

        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
