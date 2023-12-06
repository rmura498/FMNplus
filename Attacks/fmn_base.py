import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from Utils.metrics import l0_projection, l1_projection, linf_projection, l2_projection
from Utils.metrics import l0_mid_points, l1_mid_points, l2_mid_points, linf_mid_points
from Utils.loss import difference_of_logits, dlr_loss


class FMN:
    r"""
    FMN in the paper 'Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints'
    [https://arxiv.org/abs/2102.12827]

    Distance Measure : L0, L1, L2, Linf

    Args:
        model (nn.Module): The model to be attacked.
        norm (float): The norm for distance measure. Defaults to float('inf').
        steps (int): The number of steps for the attack. Defaults to 10.
        alpha_init (float): The initial alpha for the attack. Defaults to 1.0.
        alpha_final (Optional[float]): The final alpha for the attack.ha Defaults to alpha_init / 100 if not provided.
        gamma_init (float): The initial gamma for the attack. Defaults to 0.05.
        gamma_final (float): The final gamma for the attack. Defaults to 0.001.
        starting_points (Optional[Tensor]): The starting points for the attack. Defaults to None.
        binary_search_steps (int): The number of binary search steps. Defaults to 10.
    """

    def __init__(self,
                 model: nn.Module,
                 norm: float = float('inf'),
                 epsilon = None,
                 steps: int = 10,
                 alpha_init: float = 1.0,
                 alpha_final: Optional[float] = None,
                 gamma_init: float = 0.05,
                 gamma_final: float = 0.001,
                 starting_points: Optional[Tensor] = None,
                 binary_search_steps: int = 10,
                 device=torch.device('cpu'),
                 targeted=False,
                 loss='LL',
                 optimizer='SGD',
                 scheduler='CALR',
                 gradient_strategy='Normalization',
                 initialization_strategy='Standard'
                 ):
        self.model = model
        self.norm = norm
        self.steps = steps
        self.alpha_init = alpha_init
        self.alpha_final = self.alpha_init / 100 if alpha_final is None else alpha_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.starting_points = starting_points
        self.binary_search_steps = binary_search_steps
        self.device = device
        self.targeted = targeted
        self.loss = loss
        self.gradient_strategy = gradient_strategy
        self.initialization_strategy = initialization_strategy

        self.epsilon = epsilon

        optimizers = {
            'SGD': SGD,
            'Adam': Adam
        }

        schedulers = {
            'CALR': CosineAnnealingLR,
            'RLROP': ReduceLROnPlateau
        }

        self.attack_data = {
            'distance': [],
            'epsilon': [],
            'loss': []
        }

        self._dual_projection_mid_points = {
            0: (None, l0_projection, l0_mid_points),
            1: (float('inf'), l1_projection, l1_mid_points),
            2: (2, l2_projection, l2_mid_points),
            float('inf'): (1, linf_projection, linf_mid_points),
        }

        self.optimizer = optimizers[optimizer]
        self.scheduler = schedulers[scheduler]
        self.scheduler_name = scheduler

    def _gradient_update(self, delta_grad, batch_view, step_size):

        if self.gradient_strategy == 'Normalization':
            # normalize gradient
            grad_norms = delta_grad.flatten(1).norm(p=float('inf'), dim=1).clamp_(min=1e-12)
            delta_grad.div_(batch_view(grad_norms))

        elif self.gradient_strategy == 'Projection':
            oned_delta = torch.linalg.norm(delta_grad.data.flatten(1), dim=1, ord=self.norm)
            dot_product = torch.dot(oned_delta, torch.ones_like(oned_delta) * step_size)
            delta_grad = dot_product / torch.norm(delta_grad, p=self.norm) ** 2 * delta_grad

        return delta_grad

    def _initialization(self, images, labels, batch_size):

        delta = torch.zeros_like(images, device=self.device)
        is_adv = None

        if self.initialization_strategy == 'Starting Points':
            if self.starting_points is None:
                raise ValueError('Provide some starting points')
            else:
                epsilon, delta, is_adv = self._boundary_search(images, labels)
                return epsilon, delta, is_adv

        if self.norm == 0:
            epsilon = torch.ones(batch_size,
                                    device=self.device) if self.starting_points is None else \
                                    delta.flatten(1).norm(p=0,dim=0)
        else:
            epsilon = torch.full((batch_size,), float('inf'), device=self.device)

        if self.initialization_strategy == 'Random':
            delta = torch.rand(images.shape)
            delta = delta.clamp(0, 8/255)

        return epsilon, delta, is_adv

    def _boundary_search(self, images, labels):
        batch_size = len(images)
        _, _, mid_point = self._dual_projection_mid_points[self.norm]

        is_adv = self.model(self.starting_points).argmax(dim=1)
        if not is_adv.all():
            raise ValueError('Starting points are not all adversarial.')
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.ones(batch_size, device=self.device)
        for _ in range(self.binary_search_steps):
            epsilon = (lower_bound + upper_bound) / 2
            mid_points = mid_point(x0=images, x1=self.starting_points, epsilon=epsilon)
            pred_labels = self.model(mid_points).argmax(dim=1)
            is_adv = (pred_labels == labels) if self.targeted else (pred_labels != labels)
            lower_bound = torch.where(is_adv, lower_bound, epsilon)
            upper_bound = torch.where(is_adv, epsilon, upper_bound)

        delta = mid_point(x0=images, x1=self.starting_points, epsilon=epsilon) - images

        return epsilon, delta, is_adv

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        batch_size = len(images)

        dual, projection, _ = self._dual_projection_mid_points[self.norm]
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (images.ndim - 1))

        epsilon, delta, is_adv = self._initialization(images, labels, batch_size)

        _worst_norm = torch.maximum(images, 1 - images).flatten(1).norm(p=self.norm, dim=1).detach()

        init_trackers = {
            'worst_norm': _worst_norm.to(self.device),
            'best_norm': _worst_norm.clone().to(self.device),
            'best_adv': adv_images,
            'adv_found': torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        }

        multiplier = 1 if self.targeted else -1
        delta.requires_grad_(True)

        optimizer = self.optimizer([delta], lr=self.alpha_init)
        if self.scheduler_name == 'CALR':
            scheduler = self.scheduler(optimizer, T_max=self.steps)
        else:
            scheduler = self.scheduler(optimizer, factor=0.5)

        if self.epsilon is not None:
            epsilon = torch.ones(1)*self.epsilon

        for i in range(self.steps):
            optimizer.zero_grad()

            cosine = (1 + math.cos(math.pi * i / self.steps)) / 2
            gamma = self.gamma_final + (self.gamma_init - self.gamma_final) * cosine

            delta_norm = delta.data.flatten(1).norm(p=self.norm, dim=1)
            adv_images = images + delta
            adv_images = adv_images.to(self.device)

            logits = self.model(adv_images)
            pred_labels = logits.argmax(dim=1)

            if i == 0:
                labels_infhot = torch.zeros_like(logits).scatter_(
                    1,
                    labels.unsqueeze(1),
                    float('inf')
                )
                logit_diff_func = partial(
                    difference_of_logits,
                    labels=labels,
                    labels_infhot=labels_infhot
                )

            if self.epsilon is None:
                logit_diffs = logit_diff_func(logits=logits)
                ll = -(multiplier * logit_diffs)
                ll.sum().backward(retain_graph=True)
                delta_grad = delta.grad.data

            if self.loss == 'LL':
                logit_diffs = logit_diff_func(logits=logits)
                loss = -(multiplier * logit_diffs)

            elif self.loss == 'CE':
                c_loss = nn.CrossEntropyLoss(reduction='none')
                loss = -c_loss(logits, labels)

            elif self.loss == 'DLR':
                loss = -dlr_loss(logits, labels)


            is_adv = (pred_labels == labels) if self.targeted else (pred_labels != labels)
            is_smaller = delta_norm < init_trackers['best_norm']
            is_both = is_adv & is_smaller
            init_trackers['adv_found'].logical_or_(is_adv)
            init_trackers['best_norm'] = torch.where(is_both, delta_norm, init_trackers['best_norm'])
            init_trackers['best_adv'] = torch.where(batch_view(is_both), adv_images.detach(),
                                                    init_trackers['best_adv'])
            if self.epsilon is None:
                if self.norm == 0:
                    epsilon = torch.where(is_adv,
                                          torch.minimum(torch.minimum(epsilon - 1,
                                                                      (epsilon * (1 - gamma)).floor_()),
                                                        init_trackers['best_norm']),
                                          torch.maximum(epsilon + 1, (epsilon * (1 + gamma)).floor_()))
                    epsilon.clamp_(min=0)
                else:
                    distance_to_boundary = ll.detach().abs() / delta_grad.flatten(1).norm(p=dual, dim=1).clamp_(min=1e-12)
                    epsilon = torch.where(is_adv,
                                          torch.minimum(epsilon * (1 - gamma), init_trackers['best_norm']),
                                          torch.where(init_trackers['adv_found'],
                                                      epsilon * (1 + gamma),
                                                      delta_norm + distance_to_boundary)
                                          )

            loss.sum().backward()
            delta_grad = delta.grad.data

            # clip epsilon
            if self.epsilon is None:
                epsilon = torch.minimum(epsilon, init_trackers['worst_norm'])

            # gradient update strategy
            delta_grad = self._gradient_update(delta_grad, batch_view, optimizer.param_groups[0]['lr'])
            delta.grad.data = delta_grad

            optimizer.step()

            # project in place
            projection(delta=delta.data, epsilon=epsilon)
            # clamp
            delta.data.add_(images).clamp_(min=0, max=1).sub_(images)
            best_distance = torch.linalg.norm((init_trackers['best_adv'] - images).data.flatten(1),
                                              dim=1, ord=self.norm)

            if self.scheduler_name == 'RLROP':
                scheduler.step(loss.sum())
            else:
                scheduler.step()

            _epsilon = epsilon.clone()
            _distance = torch.linalg.norm((adv_images - images).data.flatten(1), dim=1, ord=self.norm)

            self.attack_data['loss'].append(loss.mean().item())
            self.attack_data['distance'].append(_distance)
            self.attack_data['epsilon'].append(_epsilon)

            if i == self.steps - 1:
                print(f"SUCCESS RATE: : {len(is_adv[is_adv == True]) * 100 / batch_size}% ")
                print(f" {len(is_adv[is_adv == True])} out of {batch_size} successfully perturbed")

        return init_trackers['best_adv'], best_distance
