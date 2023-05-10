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
# NOTE for calibration, each epoch only receive 1 batch of 64 samples
TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET, batch_size=64, shuffle=True, num_workers=16)
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


def load_density_estimators(setting: str = 'A', version: int = 0):
    setting_dir = f'output/fashion_mnist/realnvp/log/{setting}'
    models = {}
    for party in os.listdir(setting_dir):
        log.info(f'load density estimator for party {party}')
        ckpt = _find_checkpoint(os.path.join(setting_dir, party), version)
        models[party] = RealNVP.load_from_checkpoint(ckpt)
    return models


def _get_sorted_keys(d: dict):
    keys = list(d.keys())
    keys.sort()
    return keys


def deploy_global_model(setting: str = 'A', version: int = 0):
    log.info(f'Deployment Phase: {setting = }, {version = }.')
    classifiers = load_classifiers(setting, version)
    density_estimators = load_density_estimators(setting, version)
    keys = _get_sorted_keys(classifiers)
    assert keys == _get_sorted_keys(density_estimators)
    classifiers = [classifiers[k] for k in keys]
    density_estimators = [density_estimators[k] for k in keys]

    # NOTE load model, set the learning rate to 1e-5
    model = GlobalModel(classifiers, density_estimators, learning_rate=1e-5)
    logger = pl_loggers.TensorBoardLogger(
        save_dir=f'output/fashion_mnist/global/calibration/log/{setting}',
        name=f'{version}')
    trainer = pl.Trainer(logger=logger,
                         accelerator='gpu',
                         #  strategy='dp',
                         auto_select_gpus=True,
                         devices=1,
                         max_epochs=100,
                         fast_dev_run=False,
                         limit_train_batches=1)  # NOTE the calibration takes only 1 batch per epoch

    # save the zero-shot accuracy
    d, = trainer.test(model, TEST_DATALOADER)
    save_dir = f'output/fashion_mnist/global/deploy/{setting}/{version}'
    os.makedirs(save_dir, exist_ok=True)
    result = pd.DataFrame(dict_map(lambda x: [x], d))
    result.to_csv(f'{save_dir}/result.csv')

    # calibration
    trainer.fit(model, TRAIN_DATALOADER, TEST_DATALOADER)


def main():
    for version in range(20):
        for setting in 'ABCD':
            if not os.path.isdir(f'output/fashion_mnist/global/calibration/log/{setting}/{version}'):
                os.makedirs(
                    f'output/fashion_mnist/global/calibration/log/{setting}/{version}')
                deploy_global_model(setting, version)
            else:
                log.info(f'{setting=}, {version=} has been done, skip.')


# %%
if __name__ == '__main__':
    main()

# %%
