import math, pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

from ax.service.ax_client import AxClient


conf_dir = pathlib.Path('mid8_new')
# retrieve all folders inside path
confs = tuple(f for f in conf_dir.glob('*') if f.is_dir())

n = math.ceil(len(confs)/2)
fig, axes = plt.subplots(nrows=n-1, ncols=n, figsize=(16, 10), sharex=True, sharey=True)

for idx, conf in enumerate(confs):
    print(f"\nReading {conf.name} experiment...")

    # get exp json file
    exp_file = str(list(conf.glob('*.json'))[0])

    # read the experiment json file
    ax_client_restored = AxClient(verbose_logging=False).load_from_json_file(exp_file)
    model = ax_client_restored.generation_strategy.model
    ax_client_restored.get_next_trial()

    best_trial, best_params, _ = ax_client_restored.get_best_trial()
    print(f"\tBest trial is: {best_trial}\nWith best params: {best_params}\nAnd best dist: {best_params}")

    # Saving experiment report
    html_report = ax_client_restored.


    # Retrieve best trial pth for current conf
    attack_data = list(conf.glob(f'*{best_trial-1}.pth'))[0]
    print(f"\tReading this attack data file: {str(attack_data.name)}")
    attack_data = torch.load(attack_data, map_location='cpu')

    best_advs = attack_data['best_adv']
    inputs = attack_data['images']

    norms = (best_advs - inputs).flatten(1).norm(torch.inf, dim=1)

    norms_median = norms.median().item()
    robust_at_median = (norms > norms_median).float().mean().item()

    pert_sizes = torch.linspace(0, 0.2, 1000).unsqueeze(1)
    norms = (norms > pert_sizes).float().mean(dim=1)

    ax = axes.flatten()[idx]
    ax.plot(pert_sizes, norms, color='#3D5A80')
    ax.set_title(model)
    ax.grid(True)

    custom_xticks = np.linspace(0, 0.2, 5)
    ax.set_xticks(custom_xticks)

    ax.axvline(x=norms_median, color='#5DA271', linewidth=1, linestyle='--')
    ax.scatter(norms_median, robust_at_median, color='#EE6C4D', marker='*', label='Median', zorder=3, s=30)
    ax.text(norms_median + 0.01, robust_at_median, f'{norms_median:.5f}', fontsize=16, verticalalignment='center', color='#EE6C4D')
    ax.legend()


fig.tight_layout(pad=1.5)
fig.text(0.5, 0.01, r'Perturbation $||\delta^*||$', ha='center', fontsize='large')
fig.text(0.001, 0.5, 'Robust Accuracy', va='center', rotation='vertical', fontsize='large')

plt.savefig(f"tuning_mid8_comparison.pdf", bbox_inches='tight', dpi=320)