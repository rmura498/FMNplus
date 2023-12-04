import random
from datetime import datetime

import os, pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
from autoattack.autoattack import AutoAttack

from torchvision import datasets, transforms

from Attacks.fmn_base import FMN as FMN_base
from Utils.fmn_strategies import AA_fmn_loss_ce, AA_fmn_loss_dlr
from Utils.load_model import load_data

from config import MODEL_DATASET, EXP_DIRECTORY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_index(model, cifar_dataset, prev_ids=None):
    random_sample_index = random.randint(0, len(cifar_dataset) - 1)

    while prev_ids is not None and random_sample_index in prev_ids:
        random_sample_index = random.randint(0, len(cifar_dataset) - 1)

    sample, label = cifar_dataset[random_sample_index]
    sample = sample.unsqueeze(0)

    y_pred = model(sample)
    y_pred = torch.argmax(y_pred)

    misclassified = (y_pred != label)
    if misclassified:
        print("Misclassified sample, changing...")

    return random_sample_index, sample, label


def compare_AA_fmn(
        model_id=8,
        steps=20,
        epsilon=8 / 255,
        loss='LL',
        optimizer='SGD',
        scheduler='CALR',
        gradient_strategy='Normalization',
        initialization_strategy='Standard',
        norm='Linf',
        AA_attack_to_run=['apgd-ce', ],
        AA_n_restarts=5,
        AA_epsilon=8 / 255,
        exp_name='base',
        seed=42,
        prev_ids=None
):
    torch.manual_seed(seed)
    random.seed(seed)

    # model definition
    model, dataset, model_name, dataset_name = load_data(model_id=model_id)
    model.eval()
    model.to(device)

    # retrieving FMN set of configurations
    fmn_dict = AA_fmn_loss_ce()
    fmn_dict_dlr = AA_fmn_loss_dlr()

    cifar_dataset = datasets.CIFAR10(root='./Models/data', train=False, download=True, transform=transforms.ToTensor())

    random_sample_index, sample, label = get_random_index(model, cifar_dataset, prev_ids)

    label = torch.tensor([label, ])
    print(f"Sample id: {random_sample_index}")

    # running autoattack
    adversary = AutoAttack(model, norm=norm, eps=AA_epsilon, version='standard', device=device)
    adversary.attacks_to_run = AA_attack_to_run
    adversary.apgd.n_restarts = AA_n_restarts
    adversary.apgd.n_iter = steps

    adversary.run_standard_evaluation(sample, label, bs=1)

    norm_to_num = {
        'Linf': float('inf'),
        'L2': 2,
        'L1': 1,
        'L0': 0
    }

    # running FMN
    for i in range(len(fmn_dict)):
        print(i)
        print("Running:", fmn_dict[str(i)])

        attack = FMN_base(model, steps=steps, loss=fmn_dict[str(i)]['loss'], device=device, epsilon=epsilon,
                          initialization_strategy=initialization_strategy, gradient_strategy=gradient_strategy,
                          scheduler=fmn_dict[str(i)]['scheduler'], optimizer=fmn_dict[str(i)]['optimizer'],
                          norm=norm_to_num[norm], alpha_init=100, alpha_final=None)

        adv_x, best_distance = attack.forward(sample, label)
        last_dist = attack.attack_data['distance'][-1]
        print(f"Last distance:\n{last_dist}")
        attack_data = attack.attack_data
        loss_AA = adversary.apgd.loss_total[-steps:]
        loss_fmn = [loss_indiv.detach().item() for loss_indiv in attack.attack_data['loss'][-steps:]]

        loss_strat = {
            'Loss_fmn': loss_fmn,
            'Loss_AA': loss_AA
        }

        current_exp_dir = os.path.join(EXP_DIRECTORY, 'loss_AA_FMN_comparison', model_name, dataset_name,
                                       str(random_sample_index))
        if not os.path.exists(current_exp_dir): os.makedirs(current_exp_dir)
        filename = os.path.join(current_exp_dir, f"FMNvsAA-{fmn_dict[str(i)]['loss']}_{fmn_dict[str(i)]['optimizer']}_"
                                                 f"{fmn_dict[str(i)]['scheduler']}_sid_{random_sample_index}.pkl")
        # Save the object to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(loss_strat, file)

    return random_sample_index


if __name__ == '__main__':
    prev_ids = None
    for i in range(48):
        prev_id = compare_AA_fmn(epsilon=8 / 255, AA_n_restarts=1, prev_ids=prev_ids)
        if prev_ids is None: prev_ids = []
        prev_ids.append(prev_id)
