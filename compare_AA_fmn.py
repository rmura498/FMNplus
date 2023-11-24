from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from autoattack.autoattack import AutoAttack

from Attacks.fmn_base import FMN as FMN_base
from Utils.fmn_strategies import fmn_strategies
from Utils.load_model import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compare_AA_fmn(model_id, steps, batch_size, epsilon=None, loss='LL', optimizer='SGD',
                             scheduler='CALR',
                             gradient_strategy='Normalization',
                             initialization_strategy='Standard', exp_name='base',
                             norm = 'Linf', shuffle=False,
                             AA_attack_to_run=['apgd-ce',],
                             AA_n_restarts=5,
                             AA_epsilon=8/255
                   ):
    # model definition
    model, dataset, model_name, dataset_name = load_data(model_id=model_id)
    model.eval()
    model.to(device)

    # retrieving FMN set of configurations
    fmn_dict = fmn_strategies()

    # dataset definition
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle)

    samples = torch.empty([1, 3, 32, 32])
    labels = torch.empty([1], dtype=torch.int64)
    i = 0

    for sample, label in dataloader:
        y_pred = model(sample.reshape(1, 3, 32, 32))
        y_pred = torch.argmax(y_pred)

        if y_pred == label:
            samples = torch.cat([samples, sample])
            labels = torch.cat([labels, label])
            i += 1
        if i >= batch_size:
            break
    # print(samples)
    samples = samples[1:]
    labels = labels[1:]

    loss_fmn = []
    loss_AA = []

    for i in range(batch_size):
        sample = samples[i].unsqueeze(0)
        label = torch.tensor([labels[i],])

        # running autoattack
        adversary = AutoAttack(model, norm=norm, eps=AA_epsilon, version='standard', device=device)
        adversary.attacks_to_run = AA_attack_to_run
        adversary.apgd.n_restarts = AA_n_restarts
        adversary.apgd.n_iter = steps

        adversary.run_standard_evaluation(sample, label, bs=1)
        # print(len(adversary.apgd.loss_total))

        norm_to_num = {
            'Linf': float('inf'),
            'L2': 2,
            'L1': 1,
            'L0': 0
        }

        # running FMN
        strategy = fmn_dict['0']
        attack = FMN_base(model, steps=steps, loss=loss, device=device, epsilon=epsilon,
                          initialization_strategy=initialization_strategy, gradient_strategy=gradient_strategy,
                          scheduler=scheduler, optimizer=optimizer, norm=norm_to_num[norm])

        adv_x, best_distance = attack.forward(sample, label)
        last_dist = attack.attack_data['distance'][-1]
        print(f"Sample[{i}]\nLast distance:\n{last_dist}")

        # plot losses
        fig, ax = plt.subplots(figsize=(4,4))

        steps_x = np.arange(0, steps)
        ax.plot(steps_x, adversary.apgd.loss_total[-steps:], label='AA loss')
        ax.plot(steps_x, [loss_indiv.detach().item() for loss_indiv in attack.attack_data['loss'][-steps:]], label='FMN loss')

        ax.legend()

        # Get current date
        current_date = datetime.now()
        formatted_date = current_date.strftime("%d%m%y")
        fig.savefig(f"AA_FMN_loss_comparison_{formatted_date}_{i}_{exp_name}.pdf")

        loss_AA.append(adversary.apgd.loss_total[-steps:])
        loss_fmn.append(attack.attack_data['loss'][-steps:])

    return loss_fmn, loss_AA


