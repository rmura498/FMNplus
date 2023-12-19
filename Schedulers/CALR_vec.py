import torch
import math
from torch import inf
import warnings

class VectorizedCosineAnnealing:
    """Reduce learning rate when a metric has stopped improving.

    Args:
        loss (torch.Tensor): Tensor of losses.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, batch_size, total_iter, patience=2, threshold=1e-4,
                 min_step=0, eps=1e-8, verbose=False, device='cpu'):


        if batch_size is None or batch_size <= 0:
            raise ValueError('Batch Size cannot be 0.')
        self.batch_size = batch_size

        self.min_steps = torch.ones(batch_size).to(device) * min_step
        self.patience = patience
        self.threshold = threshold
        self.best = None
        self.num_bad_epochs = torch.zeros(batch_size).to(device)
        self.single_iter = torch.zeros(batch_size).to(device)
        self.eps = eps
        self.last_epoch = 0
        self.mode_worse = inf
        self.best = self.mode_worse
        self.verbose = verbose
        self.total_iter = total_iter /2

    def cosine(self, iteration):
        cosine = (1 + torch.cos(math.pi * iteration / self.total_iter)) / 2
        return cosine

    def step(self, loss, steps):
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        loss_is_better = self.is_better(loss, self.best)
        self.best = torch.where(loss_is_better, loss, self.best)
        self.num_bad_epochs = torch.where(
            loss_is_better,
            0.0,
            self.num_bad_epochs+1
        )

        patience_expired = self.num_bad_epochs > self.patience
        self.num_bad_epochs = torch.where(
            patience_expired,
            0.0,
            self.num_bad_epochs
        )

        self.single_iter = torch.where(
            patience_expired,
            self.single_iter+1,
            self.single_iter
        )

        if self.verbose:
            print(f"Num bad epochs:\n{self.num_bad_epochs}")
            print(f"Patience expired:\n{patience_expired}")
            print(f"Single Iter:\n{self.single_iter}")

        new_steps = torch.maximum(steps * self.cosine(self.single_iter), self.min_steps)
        #steps_improved = (steps - new_steps > self.eps)
        #patience_exp_and_is_better = patience_expired & steps_improved
        # patience_exp_and_is_better = patience_expired
        steps = torch.where(
            patience_expired,
            new_steps,
            steps
        )

        self.single_iter = torch.where(
            self.single_iter >= self.total_iter,
            0,
            self.single_iter
        )

        if self.verbose:
            print(f"Current steps:\n{steps}")

        return steps

    def is_better(self, cur_loss, best_loss):
        rel_epsilon = 1. - self.threshold
        return cur_loss < best_loss * rel_epsilon

