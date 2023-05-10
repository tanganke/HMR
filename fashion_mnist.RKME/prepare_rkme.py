#! /usr/bin/env python3
# %%
import os
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from RKME import reduced_KME_2
from pyutils.base import log

REDUCED_SIZE = 10
GAMMA = 0.01
# %%


def load_multiparty_features(setting: str = 'A'):
    res = {}
    setting_dir = Path(
        f'output/fashion_mnist.RKME/features.resnet101/{setting}')
    for party_dir in setting_dir.iterdir():
        if not party_dir.is_dir():
            continue
        res[party_dir.name] = torch.load((party_dir / 'feature.pt').as_posix())
    return res


# %%
def prepare_rkme(setting: str = 'A'):
    log.info(f'Uploading Phase. {setting = }.')
    features = load_multiparty_features(setting)

    for party, feature_points in features.items():
        log.info(f'{setting = }, {party = }.')
        log.info(f'size of feature tensor: {feature_points.size()}')
        rkme = reduced_KME_2(points=feature_points,
                             gamma=GAMMA,
                             reduced_size=REDUCED_SIZE)

        dataloader = DataLoader(TensorDataset(
            feature_points), batch_size=1024, shuffle=True)
        logger = pl_loggers.TensorBoardLogger(
            save_dir=f'output/fashion_mnist.RKME/features.resnet101.RKME.M=10/log/{setting}',
            name=f'{party}')
        trainer = pl.Trainer(logger=logger,
                             accelerator='gpu',
                             devices=1,
                             auto_select_gpus=True,
                             max_epochs=2_000,
                             fast_dev_run=False)
        trainer.fit(rkme, dataloader)


def main():
    for setting in 'ABCD':
        prepare_rkme(setting)


# %%
if __name__ == '__main__':
    for _ in range(20): # repeat 20 times
        main()
