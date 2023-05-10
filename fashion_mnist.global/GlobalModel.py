# %%
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Iterable
# %%


def position(l: Iterable, item):
    for idx, i in enumerate(l):
        if item == i:
            return idx
    raise Exception(f'item `{item}` not find in {l}.')


def union1d(*iters):
    if len(iters) == 0:
        return []
    s = set().union(*iters)
    s = list(s)
    s.sort()
    return s

# %%


class GlobalModel(pl.LightningModule):
    def __init__(self,
                 classifiers: Iterable[nn.Module] = None,
                 density_estimators: Iterable[nn.Module] = None,
                 learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters('learning_rate')
        self.num_parties = num_parties = len(classifiers)
        assert self.num_parties == len(density_estimators), \
            "number of classifiers must be equal to the number of density estimators"
        self.classifiers = nn.ModuleList(classifiers)
        self.density_estimators = nn.ModuleList(density_estimators)

        # classes
        self.classes = union1d(*tuple(self._classes(i)
                                      for i in range(num_parties)))
        self.class_index = [nn.Parameter(self._init_class_index(i), requires_grad=False)
                            for i in range(num_parties)]
        # NOTE nn.ParameterList with DataParallel is not supported and
        # will appear empty for the models replicated on each GPU except the original one.
        # so I just set parameters here.
        for i in range(num_parties):
            setattr(self, f'_class_index_{i}', self.class_index[i])

        # prior p(S)
        self.num_samples = sum(self._num_samples(i)
                               for i in range(num_parties))
        self.p_samples = nn.Parameter(
            torch.tensor([self._num_samples(i)
                          for i in range(num_parties)]) / self.num_samples,
            requires_grad=False)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def _init_class_index(self, i: int):
        res = torch.zeros((len(self._classes(i)),), dtype=torch.int64)
        for j in range(len(res)):
            res[j] = position(self.classes, self._classes(i)[j])
        return res

    def _classes(self, i: int):
        return self.classifiers[i].classes

    def _num_samples(self, i: int):
        return self.classifiers[i].num_samples

    def predict(self, x):
        """predict class label for input x

        Args:
            x (torch.Tensor): (batch_size, *)
            log_px_offset (int, optional): Defaults to 1.

        Returns:
            torch.Tensor: (batch_size,)
        """
        objective, _ = self(x)
        return torch.argmax(objective, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x(Tensor): (batch_size, *)

        Returns:
            Tensor: objective function
                (batch_size, num_classes)
                num_classes = len(self.classes)
            Tensor: log likelihood, p(x|S_i)
                (batch_size, num_parties, 1)
        """
        batch_size = x.size(0)
        device = x.device

        posterior = torch.zeros(
            (batch_size, self.num_parties, len(self.classes)),
            dtype=torch.float32, device=device)
        log_px = torch.zeros(
            (batch_size, self.num_parties, 1),
            dtype=torch.float32, device=device)

        # assign posterior probability & log likelihood
        # posterior:
        #       (#batch, #parties, #classes)
        # log_px:
        #       (#batch, #parties, 1)
        # px:
        #       exp(log_px - max_by_party(log_px))
        #       (#batch, #parties, 1)
        for i in range(self.num_parties):
            posterior_i = self.classifiers[i](x).softmax(-1)
            posterior[:, i, self.class_index[i]] = posterior_i
            log_px[:, i, 0] = self.density_estimators[i](x)

        max_log_px = torch.max(log_px, dim=1, keepdim=True)[0]
        log_px = log_px - max_log_px
        px = torch.exp(log_px).detach()
        # global_posterior:
        #       (#batch, #parties, #classes)
        global_posterior = posterior * px * self.p_samples.unsqueeze(1)
        objective = torch.sum(global_posterior, dim=1)
        return objective, log_px

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        device = x.device

        objective, log_px = self(x)
        self.train_acc(objective.softmax(-1).cpu(), y.cpu())

        with torch.no_grad():
            # construct y_mask
            y_mask = torch.zeros((batch_size, self.num_parties),
                                dtype=torch.float32, device=device)
            for i in range(batch_size):
                for j in range(self.num_parties):
                    _class_index = self.class_index[j].data
                    if _class_index.device != device:
                        _class_index = _class_index.to(device)
                    y_mask[i, j] = 1 if y[i] in _class_index else 0
            y_mask = y_mask.view((batch_size, self.num_parties, 1))

        obj_y = torch.take_along_dim(objective, y.view((batch_size, 1)), 1)
        mpce = -obj_y
        mpce = mpce.mean()
        loss = mpce - (log_px * y_mask).sum(dim=1).mean()

        self.log('train_acc', self.train_acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.predict(x)
        self.val_acc(y_pred.cpu(), y.cpu())
        self.log('val_acc', self.val_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.predict(x)
        self.test_acc(y_pred.cpu(), y.cpu())
        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


# %%
