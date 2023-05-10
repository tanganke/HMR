#! /usr/env/bin python3
# %%
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from torch.utils.data import DataLoader

from ConvNet import ConvNet
from data.fashion_mnist import fashion_mnist
from pyutils.base import log

# %%


def prepare_conv(setting: str = 'A'):
    log.info(f'Preparing ConvNet for classification. {setting = }.')
    for party, dataset in fashion_mnist.load_multiparty_datasets(setting).items():
        log.info(f'{party = }, dataset: {len(dataset)=}, {dataset.classes=}')
        model = ConvNet(in_channels=1, output_classes=len(dataset.classes),
                        classes=dataset.classes, num_samples=len(dataset))
        dataloader = DataLoader(dataset, batch_size=128,
                                shuffle=True, num_workers=16)

        logger = pl_loggers.TensorBoardLogger(
            save_dir=f'output/fashion_mnist/conv/log/{setting}',
            name=f'{party}')
        trainer = pl.Trainer(logger=logger,
                             accelerator='gpu',
                             auto_select_gpus=True,
                             devices=1,
                             max_epochs=20,
                             fast_dev_run=False)
        trainer.fit(model, dataloader)


def main():
    for setting in 'ABCD':
        prepare_conv(setting)


# %%
if __name__ == '__main__':
    for version in range(20):
        main()
