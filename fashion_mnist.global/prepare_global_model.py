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

from pyutils.base import log
from pyutils.base.dict import dict_map
from pyutils.base.path import listdir_fullpath
from data.fashion_mnist import fashion_mnist
from GlobalModel import GlobalModel
from ConvNet import ConvNet
from RealNVP import RealNVP


TRAIN_DATASET = fashion_mnist.load_global_datasets(train=True)
TEST_DATASET = fashion_mnist.load_global_datasets(train=False)
TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET, batch_size=128, shuffle=True, num_workers=16)
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
        ckpt = _find_checkpoint(os.path.join(setting_dir, party), version)
        # load hyper parameters from checkpoints, not model parameters
        checkpoint = torch.load(ckpt)
        hparams = checkpoint['hyper_parameters']
        # NOTE initialize untrained conv net
        models[party] = ConvNet(**hparams)
    return models


def load_density_estimators(setting: str = 'A', version: int = 0):
    setting_dir = f'output/fashion_mnist/realnvp/log/{setting}'
    models = {}
    for party in os.listdir(setting_dir):
        ckpt = _find_checkpoint(os.path.join(setting_dir, party), version)
        # load hyper parameters from checkpoints, not model parameters
        checkpoint = torch.load(ckpt)
        hparams = checkpoint['hyper_parameters']
        # NOTE initialize untrained realnvp net
        models[party] = RealNVP(**hparams)
    return models


def _get_sorted_keys(d: dict):
    keys = list(d.keys())
    keys.sort()
    return keys


def deploy_global_model(setting: str = 'A'):
    log.info(
        f'Calibrate global model from random initialization: {setting = }.')
    # NOTE only need the hyper parameters, so verison doesn't matter
    classifiers = load_classifiers(setting, version=0)
    # same as classifiers
    density_estimators = load_density_estimators(setting, version=0)
    keys = _get_sorted_keys(classifiers)
    assert keys == _get_sorted_keys(density_estimators)
    classifiers = [classifiers[k] for k in keys]
    density_estimators = [density_estimators[k] for k in keys]

    # NOTE when training from random initialization, the learning rate is set to 1e-4.
    model = GlobalModel(classifiers, density_estimators, learning_rate=1e-4)
    logger = pl_loggers.TensorBoardLogger(
        save_dir=f'output/fashion_mnist/global/raw/log/',
        name=f'{setting}')
    trainer = pl.Trainer(logger=logger,
                         accelerator='gpu',
                         #  strategy='dp',
                         auto_select_gpus=True,
                         devices=1,
                         max_epochs=20,
                         fast_dev_run=False)

    # save the very beginning accuracy
    d, = trainer.test(model, TEST_DATALOADER)

    # calibration from raw global model
    trainer.fit(model, TRAIN_DATALOADER, TEST_DATALOADER)


def main():
    for version in range(20):
        for setting in 'ABCD':
            if not os.path.exists(f'output/fashion_mnist/global/raw/log/{setting}/version_19'):
                deploy_global_model(setting)


# %%
if __name__ == '__main__':
    main()

# %%
