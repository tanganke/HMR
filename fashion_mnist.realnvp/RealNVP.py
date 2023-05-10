import os
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pyutils.base import log


class NF(nn.Module):
    """
    Normalizing flow for density estimation and sampling
    """

    def __init__(self,
                 layers,
                 loc: torch.Tensor,
                 covariance_matrix: torch.Tensor):
        R"""
        Initialize normalizing flow

        Args:
            layers: list of layers f_j with tractable inverse and log-determinant 
                (e.g., RealNVPLayer)
        """
        super(NF, self).__init__()
        self.loc = nn.Parameter(loc, requires_grad=False)
        self.covariance_matrix = nn.Parameter(
            covariance_matrix, requires_grad=False)
        self.layers = layers

    @property
    def prior(self):
        """
        latent distribution, 
            e.g., distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
        """
        return torch.distributions.MultivariateNormal(self.loc, self.covariance_matrix)

    def g(self, z):
        """
        Args:
            z: latent variable

        Returns:
            g(z) and hidden states
        """
        y = z
        ys = [torch.clone(y).detach()]
        for i in range(len(self.layers)):
            y, _ = self.layers[i].f(y)
            ys.append(torch.clone(y).detach())

        return y, ys

    def ginv(self, x):
        R"""
        Args:
            x: sample from dataset

        Returns:
            g^(-1)(x), value of log-determinant, and hidden layers
        """
        p = x
        log_det_ginv = torch.zeros(x.shape[0])
        if x.is_cuda:
            log_det_ginv = log_det_ginv.cuda()
        ps = [torch.clone(p).detach()]
        for i in reversed(range(len(self.layers))):
            p, log_det_finv = self.layers[i].finv(p)
            ps.append(torch.clone(p).detach())
            log_det_ginv += log_det_finv

        return p, log_det_ginv, ps

    def log_prob(self, x) -> torch.Tensor:
        R"""
        Compute log-probability of a sample using change of variable formula

        Args:
            x: sample from dataset
        Returns:
            Tensor: logp_{\theta}(x)
        """
        z, log_det_ginv, _ = self.ginv(x)
        log_pz = self.prior.log_prob(z)
        if x.is_cuda:
            log_pz = log_pz.cuda()
        return log_pz + log_det_ginv

    def score_samples(self, x) -> np.ndarray:
        if len(x.size()) > 1:
            x = torch.flatten(x, 1)
        ans = self.log_prob(x)
        return ans

    def sample(self, s):
        R"""
        Draw random samples from p_{\theta}

        Args:
            s: number of samples to draw
        """
        z = self.prior.sample((s, 1)).squeeze(1)
        x, _ = self.g(z)
        return x

    def fit(self, dataloader, epoch=128, lr=1e-4, weight_decay=1e-2, cuda=False):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)
        epoch_loss_his = []

        for epoch_id in range(epoch):
            epoch_loss = 0
            epoch_size = 0
            for batch_id, (x, _) in enumerate(dataloader):
                if cuda:
                    x: torch.Tensor = x.cuda()
                x = torch.flatten(x, 1)
                batch_size = x.size()[0]
                epoch_size += batch_size

                loss = -self.log_prob(x).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_size

                print(f'epoch/batch: {epoch_id}/{batch_id}, loss: {loss}')
            epoch_loss /= epoch_size
            print(f'epoch: {epoch_id}, epoch_loss: {epoch_loss}')
            epoch_loss_his.append(epoch_loss)
        return pd.DataFrame({'epoch_loss': epoch_loss_his})


class RealNVPLayer(nn.Module):
    """
    Real non-volume preserving flow layer

    Reference: Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016, May 27).
               Density estimation using Real NVP. arXiv.org.
    """

    def __init__(self, s, t, mask):
        """
        Initialize real NVP layer
        :param s: network to compute the shift
        :param t: network to compute the translation
        :param mask: splits the feature vector into two parts
        """
        super(RealNVPLayer, self).__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = t
        self.s = s

    def f(self, y):
        """
        apply the layer function f

        Args:
            y:  feature vector
        """
        y1 = y * self.mask
        s = self.s(y1)
        t = self.t(y1)
        y2 = (y * torch.exp(s) + t) * (1 - self.mask)
        return y1 + y2, torch.sum(s, dim=1)

    def finv(self, y):
        """
        apply the inverse of the layer function

        Args:
            y: feature vector
        """
        y1 = self.mask * y
        s = self.s(y1)
        t = self.t(y1)
        y2 = (1 - self.mask) * (y - t) * torch.exp(-s)
        return y1 + y2, -torch.sum(s, dim=1)


class RealNVP(pl.LightningModule):
    def __init__(self, num_layers=12, width=512, learning_rate=1e-4):
        """
        Get a NVP network for MNIST
        """
        # layers and masks
        super().__init__()
        self.save_hyperparameters()
        log.info(f'Constructing `RealNVP`, hparams =\n{self.hparams}.')
        w = 28
        layers = torch.nn.ModuleList()
        for k in range(num_layers):
            mask = torch.tensor(
                [1 - (k % 2)] * (w * w // 2) + [k % 2] * (w * w // 2))
            t = nn.Sequential(nn.Linear(w * w, width),
                              nn.LeakyReLU(),
                              nn.Linear(width, width),
                              nn.LeakyReLU(),
                              nn.Linear(width, w * w),
                              nn.Tanh())
            s = nn.Sequential(nn.Linear(w * w, width),
                              nn.LeakyReLU(),
                              nn.Linear(width, width),
                              nn.LeakyReLU(),
                              nn.Linear(width, w * w),
                              nn.Tanh())
            layer = RealNVPLayer(s, t, mask)
            layers.append(layer)

        self.model = NF(layers, loc=torch.zeros(w * w),
                        covariance_matrix=torch.eye(w * w))

    # NOTE necessary
    def forward(self, x):
        """
        Returns:
            Tensor: log likelihood estimation
        """
        return self.model.score_samples(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x = torch.flatten(x, 1)
        loss = -self.model.log_prob(x).mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
