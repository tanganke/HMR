#! /usr/bin/env python3
# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torchmetrics
from typing import Iterable

from pyutils.base import log
from pyutils.base.path import listdir_fullpath
from pyutils.torch import to_device
from data.fashion_mnist import fashion_mnist
from ConvNet import ConvNet

TEST_DATASET = fashion_mnist.load_global_datasets(train=False)
TEST_DATALOADER = DataLoader(
    TEST_DATASET, batch_size=128, shuffle=False, num_workers=16)
# %%


def _find_checkpoint(party_dir: str, version: int):
    ckpt_dir = os.path.join(party_dir, f'version_{version}', 'checkpoints')
    ckpt_file = listdir_fullpath(ckpt_dir)[0]
    assert ckpt_file.split('.')[-1] == 'ckpt'
    return ckpt_file


def load_classifiers(setting: str = 'A', version: int = 0):
    setting_dir = f'output/fashion_mnist/conv/log/{setting}'
    models = {}
    for party in os.listdir(setting_dir):
        log.info(f'load classifier for party {party}')
        ckpt = _find_checkpoint(os.path.join(setting_dir, party), version)
        models[party] = ConvNet.load_from_checkpoint(ckpt)
    return models


def position(l: Iterable, item):
    for idx, i in enumerate(l):
        if item == i:
            return idx
    raise Exception(f'item `{item}` not find in {l}.')


def union1d(*iters):
    if len(iters) == 0:
        return []
    s = set().union(*iters)
    s = list(s)
    s.sort()
    return s

# %%


class MaxPredictor(pl.LightningModule):
    def __init__(self, classifiers=None):
        super().__init__()
        self.num_parties = num_parties = len(classifiers)
        self.classifiers = nn.ModuleList(classifiers)

        # classes
        self.classes = union1d(*tuple(self._classes(i)
                                      for i in range(num_parties)))
        self.class_index = nn.ParameterList([nn.Parameter(self._init_class_index(i), requires_grad=False)
                                             for i in range(num_parties)])

        self.test_acc = torchmetrics.Accuracy()

    def _init_class_index(self, i: int):
        res = torch.zeros((len(self._classes(i)),), dtype=torch.int64)
        for j in range(len(res)):
            res[j] = position(self.classes, self._classes(i)[j])
        return res

    def _classes(self, i: int):
        return self.classifiers[i].classes

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        posterior = torch.zeros(
            (batch_size, self.num_parties, len(self.classes)),
            dtype=torch.float32, device=device)

        for i in range(self.num_parties):
            class_index = to_device(self.class_index[i], x)
            posterior[:, i, class_index] = self.classifiers[i](x).softmax(-1)

        return posterior.max(1).values

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_pred = logits.argmax(-1)
        self.test_acc(y_pred.cpu(), y.cpu())
        self.log('test_acc', self.test_acc)

# %%


def main():
    for version in range(20):
        for setting in 'ABCD':
            if os.path.isdir(f'output/fashion_mnist.MAX/log/{setting}/{version}'):
                log.info(f'{setting=}, {version=} done, skip.')
                continue
            else:
                os.makedirs(
                    f'output/fashion_mnist.MAX/log/{setting}/{version}')

            # load classifiers
            classifiers = load_classifiers(setting, version).values()
            # contruct max predictor model
            model = MaxPredictor(classifiers)

            logger = pl_loggers.TensorBoardLogger(
                save_dir=f'output/fashion_mnist.MAX/log/{setting}',
                name=f'{version}')
            trainer = pl.Trainer(logger=logger,
                                 accelerator='cpu',
                                 #  strategy='dp',
                                 auto_select_gpus=True,
                                 devices=1,
                                 max_epochs=100,
                                 fast_dev_run=False,
                                 limit_train_batches=1)

            trainer.test(model, TEST_DATALOADER)


# %%
if __name__ == '__main__':
    main()
# %%
