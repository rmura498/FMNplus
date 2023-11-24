import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from autoattack.autoattack import AutoAttack

from Attacks.fmn_base import FMN as FMN_base
from Attacks.fmn_base_saliency_test import FMN as FMN_saliency
from Utils.fmn_strategies import fmn_strategies
from Utils.load_model import load_data

parser = argparse.ArgumentParser(description='Perform multiple attacks using FMN+, a parametric version of FMN.')
parser.add_argument('-mid', '--model_id',
                    default=0,
                    help='Model id to test.')
parser.add_argument('-s', '--steps',
                    default=30,
                    help='Provide the number of steps of a single attack.')
parser.add_argument('-bs', '--batch_size',
                    default=10,
                    help='Provide the batch size.')
parser.add_argument('-eps', '--epsilon',
                    default=None,
                    help='Provide epsilon value.')

parser.add_argument('-en', '--exp_name', default='base')

args = parser.parse_args()

model_id = int(args.model_id)
steps = int(args.steps)
batch_size = int(args.batch_size)

if args.epsilon is not None:
    epsilon = float(args.epsilon)
else:
    epsilon = None

exp_name = args.exp_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model definition
model, dataset, model_name, dataset_name = load_data(model_id=model_id)
model.eval()
model.to(device)

# retrieving FMN set of configurations
fmn_dict = fmn_strategies()

# dataset definition
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

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

for i in range(batch_size):
    sample = samples[i].unsqueeze(0)
    label = torch.tensor([labels[i],])

    # running autoattack
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', device=device)
    adversary.attacks_to_run = ['apgd-ce',]
    adversary.apgd.n_restarts = 5
    adversary.apgd.n_iter = steps

    adversary.run_standard_evaluation(sample, label, bs=1)
    # print(len(adversary.apgd.loss_total))

    # running FMN
    strategy = fmn_dict['0']
    attack = FMN_base(model, steps=steps, loss='LL', device=device)

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
    fig.savefig(f"AA_FMN_loss_comparison_{formatted_date}_{i}.pdf")

