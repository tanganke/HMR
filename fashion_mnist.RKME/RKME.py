# %%
from functools import cached_property
import re
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pyutils.base import log
# %%


def gaussian_kernel(x1: Tensor, x2: Tensor, gamma=0.01) -> Tensor:
    """
    gaussian_kernel

    Args:
        x1 (Tensor): (data_dim, )
        x2 (Tensor): (data_dim, )
        gamma (int, optional): Defaults to 1.

    Returns:
        Tensor: (data_dim, )
    """
    return torch.exp(-gamma * torch.norm(x1 - x2, dim=-1))


def gaussian_kernel_matrix(x1: Tensor, x2: Tensor, gamma=0.01) -> Tensor:
    """

    Args:
        x1 (Tensor): (N, data_dim)
        x2 (Tensor): (M, data_dim)
        gamma (int, optional): Defaults to 1.

    Returns:
        Tensor: (N, M)
    """
    assert x1.dim() == x2.dim() == 2
    x1 = x1.unsqueeze(1)
    return torch.exp(-gamma * torch.norm(x1 - x2, dim=-1))


def gather_data(dataloader: DataLoader) -> Tensor:
    """
    gather data samples of dataloader, return a numpy array

    Args:
        dataloader (DataLoader)

    Returns:
        Tensor
    """
    return torch.concat([batch[0] for batch in dataloader])


class empirical_KME(pl.LightningModule):
    """
    Empirical Kernel Mean Embedding

    ref: Model Reuse with Reduced Kernel Mean Embedding Specification, eq.(3)

    >>> ekme = empirical_KME(points)
    >>> ekme(x)

    """

    def __init__(self, points: Tensor, gamma=0.01):
        """

        Args:
            dataloader (DataLoader): #* not shuffled
        """
        super().__init__()
        assert points.dim() == 2
        self.points = nn.Parameter(points, requires_grad=False)
        self.num_samples = self.points.size(0)
        self.data_dim = self.points.size(1)
        self.gamma = gamma

    @cached_property
    def _gaussian_kernel_matrix(self) -> Tensor:
        """
        return the Gram matrix of gaussian kernel

        Returns:
            Tensor: (N, N) N is the number of samples in dataloader
        """
        x = self.points
        return gaussian_kernel_matrix(x, x, gamma=self.gamma)

    def _gaussian_kernel_sum(self, x1: Tensor, x2: Tensor, gamma=0.01) -> Tensor:
        """
        gaussian kernel sum

        Args:
            x1 (Tensor): (N1, data_dim)
            x2 (Tensor): (N2, data_dim)
            gamma (int, optional): Defaults to 1.

        Returns:
            Tensor: (N2,)
        """
        res = 0
        x2 = x2.unsqueeze(1)  # (N2, 1, data_dim)
        x = x1 - x2  # (N2, N1, data_dim)
        res += torch.exp(-gamma * torch.norm(x, dim=-1)).sum(-1)
        return res

    def forward(self, x: Tensor) -> Tensor:
        """
        compute empirical KME

        Args:
            x (Tensor): a batch of samples (batch_size, data_dim)

        Returns:
            Tensor: (batch_size,)
        """

        res = self._gaussian_kernel_sum(self.points, x, gamma=self.gamma) / self.points.size(0)
        return res

# %%


class reduced_KME(pl.LightningModule):
    """
    Examples:

    >>> points = torch.randn(1000, 5)
    >>> ekme = empirical_KME(points)
    >>> rkme = reduced_KME(reduced_size=5, ekme=ekme)

    >>> trainer = pl.Trainer(accelerator='cpu', max_epochs=1, fast_dev_run=False)
    >>> trainer.fit(rkme, DummyDataLoader(10000))

    """

    def __init__(self, reduced_size: int = None, learning_rate=1e-4, ekme: empirical_KME = None):
        super().__init__()
        self.reduced_size = reduced_size
        if ekme is not None:
            self.ekme = ekme  # don't want to save ekme
            self.save_hyperparameters({'reduced_size': reduced_size,
                                       'learning_rate': learning_rate,
                                       'data_dim': ekme.data_dim,
                                       'num_samples': ekme.points.size(0),
                                       'gamma': ekme.gamma})
            self.reduced_points = \
                nn.Parameter(self._kmean_init_points(ekme), requires_grad=True)
            self.betas = \
                nn.Parameter(torch.randn(reduced_size), requires_grad=True)

    def _kmean_init_points(self, ekme: empirical_KME) -> Tensor:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.reduced_size).fit(ekme.points)
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    @property
    def gamma(self):
        return self.hparams.gamma

    def forward(self, x: Tensor) -> Tensor:
        """
        compute reduced KME

        Args:
            x (Tensor): a batch of samples (batch_size, data_dim)

        Returns:
            Tensor: (batch_size,)
        """
        res = gaussian_kernel_matrix(self.reduced_points, x, gamma=self.gamma)  # (reduced_size, batch_size)
        res = (self.betas.unsqueeze(1) * res).sum(0)
        return res

    def training_step(self, batch, batch_idx):
        if not hasattr(self, 'ekme'):
            raise RuntimeError("This model has already been trained.")

        Kx: Tensor = self.ekme._gaussian_kernel_matrix
        Kx.to(self.device)
        Kz = gaussian_kernel_matrix(self.reduced_points, self.reduced_points, gamma=self.gamma)
        F = Kx.mean() + \
            (self.betas.unsqueeze(1) * self.betas * Kz).sum() \
            - 2 * torch.dot(self.betas, self.ekme(self.reduced_points))
        self.log('F(beta,Z)', F)
        return F

    def on_save_checkpoint(self, checkpoint) -> None:
        state_dict = checkpoint['state_dict']
        # don't want to save ekme
        for k in state_dict.keys():
            if re.match('ekme.*', k) is not None:
                state_dict.pop(k)

    def on_load_checkpoint(self, checkpoint) -> None:
        self.hparams.update(checkpoint['hyper_parameters'])

        # load parameters
        state_dict = checkpoint['state_dict']
        self.reduced_points = nn.Parameter(state_dict['reduced_points'])
        self.betas = nn.Parameter(state_dict['betas'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def DummyDataLoader(steps=1):
    return DataLoader([0 for _ in range(steps)], batch_size=1, num_workers=1)


# %%
if __name__ == '__main__':
    NUM_POINTS = 100
    DATA_DIM = 5

    points = torch.randn(NUM_POINTS, DATA_DIM)
    ekme = empirical_KME(points)
    rkme = reduced_KME(reduced_size=5, ekme=ekme)

    trainer = pl.Trainer(accelerator='gpu', devices=1,
                         auto_select_gpus=True, max_epochs=1, fast_dev_run=False)
    # trainer = pl.Trainer(accelerator='cpu', max_epochs=1, fast_dev_run=False)
    trainer.fit(rkme, DummyDataLoader(100000))

# %%


class reduced_KME_2(pl.LightningModule):
    """
    Reduced Kernel Mean Embedding

    """

    def __init__(self,
                 points: Tensor = None,
                 gamma: float = 0.01,
                 reduced_size: int = 10,
                 learning_rate: float = 1e-4):
        super().__init__()
        if points is not None:
            assert points.dim() == 2
            self.points = points
            self.save_hyperparameters({'reduced_size': reduced_size,
                                       'learning_rate': learning_rate,
                                       'data_dim': points.size(1),
                                       'num_samples': points.size(0),
                                       'gamma': gamma})

            self.reduced_points = \
                nn.Parameter(self._kmean_init_points(points), requires_grad=True)
            self.betas = nn.Parameter(self._init_betas(points), requires_grad=False)

    def _kmean_init_points(self, points: Tensor) -> Tensor:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.reduced_size).fit(points)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return centers

    def _init_betas(self, points: Tensor) -> Tensor:
        with torch.no_grad():
            K = gaussian_kernel_matrix(self.reduced_points, self.reduced_points, self.gamma)
            C = gaussian_kernel_matrix(points, self.reduced_points, self.gamma)\
                .mean(0, keepdim=False).unsqueeze(1)
            betas = (torch.linalg.inv(K) @ C).squeeze()
        return betas

    def on_load_checkpoint(self, checkpoint) -> None:
        self.hparams.update(checkpoint['hyper_parameters'])

        # load parameters
        state_dict = checkpoint['state_dict']
        self.reduced_points = nn.Parameter(state_dict['reduced_points'], requires_grad=False)
        self.betas = nn.Parameter(state_dict['betas'], requires_grad=False)

    @property
    def reduced_size(self):
        return self.hparams.reduced_size

    @property
    def data_dim(self):
        return self.hparams.data_dim

    @property
    def gamma(self):
        return self.hparams.gamma

    def forward(self, x: Tensor) -> Tensor:
        """
        compute reduced KME

        Args:
            x (Tensor): a batch of samples (batch_size, data_dim)

        Returns:
            Tensor: (batch_size,)
        """
        res = gaussian_kernel_matrix(self.reduced_points, x, gamma=self.gamma)  # (reduced_size, batch_size)
        res = (self.betas.unsqueeze(1) * res).sum(0)
        return res

    def training_step(self, batch, batch_idx):
        x = batch[0]

        ekme = empirical_KME(x, gamma=self.gamma)
        Kx_mean = gaussian_kernel_matrix(x, x, gamma=self.gamma).mean()
        Kz = gaussian_kernel_matrix(self.reduced_points, self.reduced_points, gamma=self.gamma)
        F = Kx_mean + \
            (self.betas.unsqueeze(1) * self.betas * Kz).sum() \
            - 2 * torch.dot(self.betas, ekme(self.reduced_points))

        if not self.points.device == self.betas.data.device:
            self.points = self.points.to(self.betas.data.device)
        self.betas.data = self._init_betas(self.points.to(self.betas.data.device))\
            .to(self.betas.data.device)
        self.log('F(beta,Z)', F)
        return F

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
        return [optimizer], [lr_scheduler]
