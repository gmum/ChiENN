import math
from tqdm import tqdm
from typing import List, Union
from argparse import Namespace
import numpy as np

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import roc_auc_score


def train(model, loader, optimizer, loss, stdzer, device, scheduler, task):
    model.train()
    loss_all, correct = 0, 0

    for data in tqdm(loader, total=len(loader)):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        result = loss(out, stdzer(data.y))
        result.backward()

        optimizer.step()
        scheduler.step()
        loss_all += loss(stdzer(out, rev=True), data.y)

        if task == 'classification':
            predicted = torch.round(out.data)
            correct += (predicted == data.y).sum().double()

    if task == 'regression':
        return math.sqrt(loss_all / len(loader.dataset)), None  # rmse
    elif task == 'classification':
        return loss_all / len(loader.dataset), correct / len(loader.dataset)


def eval(model, loader, loss, stdzer, device, task):
    model.eval()
    error, correct = 0, 0

    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            data = data.to(device)
            out = model(data)
            error += loss(stdzer(out, rev=True), data.y).item()

            if task == 'classification':
                predicted = torch.round(out.data)
                correct += (predicted == data.y).sum().double()

    if task == 'regression':
        return math.sqrt(error / len(loader.dataset)), None  # rmse
    elif task == 'classification':
        return error / len(loader.dataset), correct / len(loader.dataset)


def test(model, loader, loss, stdzer, device, task):
    model.eval()
    error, correct = 0, 0

    preds, ys = [], []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            data = data.to(device)
            out = model(data)
            pred = stdzer(out, rev=True)
            error += loss(pred, data.y).item()
            preds.extend(pred.cpu().detach().tolist())

            if task == 'classification':
                predicted = torch.round(out.data)
                correct += (predicted == data.y).sum().double()
                ys.extend(data.y.cpu().detach().tolist())

    if task == 'regression':
        return preds, math.sqrt(error / len(loader.dataset)), None, None  # rmse
    elif task == 'classification':
        auc = roc_auc_score(ys, preds)
        return preds, error / len(loader.dataset), correct / len(loader.dataset), auc


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def build_lr_scheduler(optimizer: Optimizer, args: Namespace, train_data_size: int) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param train_data_size: The size of the training dataset.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.n_epochs],
        steps_per_epoch=train_data_size // args.batch_size,
        init_lr=[args.lr / 10],
        max_lr=[args.lr],
        final_lr=[args.lr / 10]
    )