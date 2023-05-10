#! /usr/bin/env python3
# %%
import os
import re
from typing import Iterable
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torchmetrics

from pyutils.base import log
from pyutils.base.dict import dict_map
from pyutils.base.path import listdir_fullpath
from data.fashion_mnist import fashion_mnist
from RKME import reduced_KME_2, gaussian_kernel_matrix
from ConvNet import ConvNet

TEST_DATASET = fashion_mnist.load_global_datasets(train=False)
TEST_FEATURE_POINTS = torch.load(
    'output/fashion_mnist.RKME/features.resnet101/test/feature.pt')
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
        models[party] = ConvNet.load_from_checkpoint(ckpt)
    return models


def load_RKMEs(setting: str = 'A', version: int = 0):
    setting_dir = f'output/fashion_mnist.RKME/features.resnet101.RKME.M=10/log/{setting}'
    models = {}
    for party in os.listdir(setting_dir):
        ckpt = _find_checkpoint(os.path.join(setting_dir, party), version)
        models[party] = reduced_KME_2.load_from_checkpoint(ckpt)
    return models


def _gaussian_kernel_matrix(x1, x2, gamma):
    return torch.exp(-gamma * torch.norm(x1 - x2, dim=-1))


def compute_mixture_weights(RKMEs: Iterable[reduced_KME_2], points: Tensor) -> Tensor:
    c = len(RKMEs)
    M = RKMEs[0].reduced_size
    d = RKMEs[0].data_dim
    with torch.no_grad():
        C = torch.concat(tuple(rkme(points).mean(0, keepdim=True).unsqueeze(0)
                               for rkme in RKMEs))  # (c, 1)
        Z = torch.concat(tuple(rkme.reduced_points.unsqueeze(0)
                               for rkme in RKMEs))  # (c, M, d)
        H = _gaussian_kernel_matrix(Z.view(c, 1, M, 1, d),
                                    Z.view(1, c, 1, M, d),
                                    gamma=RKMEs[0].gamma)\
            .mean((-1, -2))  # (c, c)

        weights: Tensor = torch.linalg.inv(H) @ C
    if weights.min() < 0:
        log.warn(f"Weights `{weights}` must be all positive.")
        weights = F.relu(weights)
    return weights.squeeze()


def get_sampler(weights: Tensor):
    assert weights.dim() == 1
    from torch.distributions.categorical import Categorical
    dist = Categorical(weights / weights.sum())
    return dist

# %%


class RKME_GlobalModel(pl.LightningModule):
    def __init__(self, classifiers, RKMEs):
        super().__init__()
        self.save_hyperparameters()
        self.classifiers = classifiers
        self.RKMEs = RKMEs


def _get_sorted_keys(d: dict):
    keys = list(d.keys())
    keys.sort()
    return keys


def generate_samples(mixture_weights: Tensor, RKMEs: list[reduced_KME_2], num_samples=50):
    mixture_weights_sampler = get_sampler(mixture_weights)
    S: Tensor = None
    Si = []

    for T in tqdm(range(num_samples), desc='generating samples'):
        i = mixture_weights_sampler.sample().item()
        rkme = RKMEs[i]
        if T == 0:
            x = TEST_FEATURE_POINTS[rkme(TEST_FEATURE_POINTS).argmax()]
            S = x.unsqueeze(0)
        else:
            x = TEST_FEATURE_POINTS[
                (rkme(TEST_FEATURE_POINTS)
                    - (1 / (T + 1)) * gaussian_kernel_matrix(S, TEST_FEATURE_POINTS).sum(0))
                .argmax()]
            S = torch.concat((S, x.unsqueeze(0)))
        Si.append(i)
    Si = torch.tensor(Si, dtype=torch.int32)
    return S, Si


def deploy_rkme(setting: str = 'A', version: int = 0):
    log.info(f'[RKME] Deployment phase: {setting = }, {version = }.')
    classifiers = load_classifiers(setting, version=version)
    RKMEs = load_RKMEs(setting, version)
    keys = _get_sorted_keys(classifiers)
    assert keys == _get_sorted_keys(RKMEs)
    classifiers = [classifiers[k] for k in keys]
    RKMEs = [RKMEs[k] for k in keys]

    mixture_weights = compute_mixture_weights(RKMEs, TEST_FEATURE_POINTS)
    samples = generate_samples(mixture_weights, RKMEs, 100)

    # train selector
    from sklearn.svm import SVC
    selector = SVC()
    selector.fit(*samples)

    correct = 0
    error = 0
    for point_idx, feature in tqdm(enumerate(TEST_FEATURE_POINTS),
                                   desc='predicting',
                                   total=len(TEST_FEATURE_POINTS)):
        i = selector.predict(feature.unsqueeze(0)).item()
        x, y = TEST_DATASET[point_idx]
        logits = classifiers[i](x.unsqueeze(0))
        if classifiers[i].classes[logits.argmax()] == TEST_DATASET.classes[y]:
            correct += 1
        else:
            error += 1

    return {'acc': correct / (correct + error),
            'correct': correct,
            'error': error}


def main():
    for version in range(20):
        for setting in 'ABCD':
            save_dir = f'output/fashion_mnist.RKME/features.resnet101.RKME.M=10/deploy/{setting}/version_{version}'
            if os.path.isdir(save_dir):
                log.info(f'save directory for `{setting=}, {version=}` exists, skip.')
                continue
            else:
                os.makedirs(save_dir)

            d = deploy_rkme(setting, version)
            result = pd.DataFrame(dict_map(lambda x: [x], d))
            log.info(f'{result}')
            result.to_csv(
                f'{save_dir}/result.csv')


# %%
if __name__ == '__main__':
    main()

# %%
