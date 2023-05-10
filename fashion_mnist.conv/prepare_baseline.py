#! /usr/env/bin python3
# %%
from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from torch.utils.data import DataLoader

from ConvNet import ConvNet
from data.fashion_mnist import fashion_mnist
from pyutils.base.dict import dict_map

TRAIN_DATASET = fashion_mnist.load_global_datasets(train=True)
TEST_DATASET = fashion_mnist.load_global_datasets(train=False)
TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET, batch_size=128, shuffle=True, num_workers=16)
TEST_DATALOADER = DataLoader(
    TEST_DATASET, batch_size=128, shuffle=False, num_workers=16)
# %%


def train_centralized_baseline():
    dataset = TRAIN_DATASET
    dataloader = TRAIN_DATALOADER
    model = ConvNet(in_channels=1, output_classes=len(dataset.classes),
                    classes=dataset.classes, num_samples=len(dataset))

    logger = pl_loggers.TensorBoardLogger(
        save_dir=f'output/fashion_mnist/baseline/', name='log')
    trainer = pl.Trainer(logger=logger,
                         accelerator='gpu',
                         auto_select_gpus=True,
                         devices=1,
                         max_epochs=20,
                         fast_dev_run=False)

    trainer.fit(model, dataloader)

    d, = trainer.test(model, TEST_DATALOADER)
    result = pd.DataFrame(dict_map(lambda x: [x], d))
    return result


# %%
if __name__ == '__main__':
    result = pd.DataFrame()
    for version in tqdm(range(20)):
        result = result.append(train_centralized_baseline())
        print(result)
    result.to_csv('output/fashion_mnist/baseline/result.csv')

# %%
