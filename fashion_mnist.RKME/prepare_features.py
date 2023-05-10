#! /usr/bin/env python3
"""
prepare feature vectors by pretrained resnet 101
"""

# %%
import os
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from functools import cache

from data.fashion_mnist import fashion_mnist

CUDA = torch.cuda.is_available()

# %%


@cache
def load_feature_exactor():
    model = models.resnet101(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    if CUDA:
        model.cuda()
    return model


def concat_feature_from_dataloader(dataloader: DataLoader):
    feature_exactor = load_feature_exactor()
    transform = transforms.Resize((224, 224))

    def gen_feature():
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='exacting feature'):
                x = batch[0]
                if CUDA:
                    x = x.cuda()

                x: Tensor = transform(x)
                x = x.repeat(1, 3, 1, 1)
                feature: Tensor = feature_exactor(x)
                feature = feature.cpu().detach()
                yield feature

    return torch.concat(tuple(gen_feature()))


def concat_feature_from_dataset(dataset: Dataset, batch_size=128, num_workers=16, **kwargs):
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, **kwargs)
    return concat_feature_from_dataloader(dataloader)


# %%
# NOTE compute features for multiparty settings
if __name__ == '__main__':
    for setting in 'ABCD':
        datasets = fashion_mnist.load_multiparty_datasets(setting)

        for party, dataset in datasets.items():
            save_dir = f'output/fashion_mnist.RKME/features.resnet101/{setting}/{party}'
            os.makedirs(save_dir, exist_ok=True)
            features = concat_feature_from_dataset(dataset, batch_size=128)
            torch.save(features, f'{save_dir}/feature.pt')

# %%
# NOTE compute features for test dataset
if __name__ == '__main__':
    save_dir = f'output/fashion_mnist.RKME/features.resnet101/test'
    os.makedirs(save_dir, exist_ok=True)
    dataset = fashion_mnist.load_global_datasets(train=False)
    features = concat_feature_from_dataset(dataset, batch_size=128)
    torch.save(features, f'{save_dir}/feature.pt')

# %%
