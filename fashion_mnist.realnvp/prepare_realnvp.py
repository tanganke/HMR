#! /usr/bin/env python3
# %%
import os
from torch.utils.data import DataLoader, Dataset
from typing import Dict, OrderedDict
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from pyutils.base import log
from data.fashion_mnist import fashion_mnist
from RealNVP import RealNVP


def prepare_realnvp(setting: str = 'A'):
    for party, dataset in fashion_mnist.load_multiparty_datasets(setting).items():
        if os.path.exists(f'output/fashion_mnist/realnvp/log/{setting}/{party}/version_19'):
            continue

        log.info(f'Preparing RealNVP for density estimation. {setting = }.')
        log.info(f'{party = }, dataset: {len(dataset)=}, {dataset.classes=}')
        model = RealNVP(num_layers=12, width=128, learning_rate=1e-4)
        dataloader = DataLoader(dataset, batch_size=128,
                                shuffle=True, num_workers=16)
        logger = pl_loggers.TensorBoardLogger(
            save_dir=f'output/fashion_mnist/realnvp/log/{setting}',
            name=f'{party}')
        trainer = pl.Trainer(logger=logger,
                             accelerator='gpu',
                             strategy='dp',
                             auto_select_gpus=True,
                             devices=1,
                             max_epochs=20,
                             fast_dev_run=False)
        trainer.fit(model, dataloader)


def main():
    for version in range(20):
        for setting in 'ABCD':
            prepare_realnvp(setting)


# %%
if __name__ == '__main__':
    main()
