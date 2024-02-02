import os, pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

aa_acc = [70.69, 67.31, 66.10, 64.58, 63.38, 63.35, 62.79, 61.04, 58.50]

data_dir = pathlib.Path(f'./Experiments/baseline_attacks')
models = tuple({f for f in data_dir.glob('*') if (f.is_dir() and 'mid' in f.name)})
models = sorted(models)

print(models)

# change this values
nrows = 3
ncols = 3
figure, axes = plt.subplots(nrows, ncols, figsize=(20,16), sharex=True, sharey=True)

for idx, model in enumerate(models):
    batches = [f for f in model.glob('*')]
    best_dist = []
    best_advs = None
    inputs = None
    for batch in batches[:len(batches)]:
        attack_data = torch.load(batch, map_location='cpu')

        _best_advs = attack_data['best_adv']
        _inputs = attack_data['images']

        best_advs = _best_advs if best_advs is None else torch.cat([best_advs, _best_advs])
        inputs = _inputs if inputs is None else torch.cat([inputs, _inputs])
        best_dist.append(torch.linalg.norm((_best_advs-_inputs), ord=torch.inf, dim=1).median().item())
    best_dist = torch.tensor(best_dist).mean().item()
    print(model.name.split('_')[2])
    print(f'BEST MENA DISTANCE:{best_dist}')
    n_samples = best_advs.shape[0]
    norms = (best_advs - inputs).flatten(1).norm(torch.inf, dim=1)

    rob_acc = (norms>8/255).float().mean()
    acc = (norms>0).float().mean()

    pert_sizes = torch.linspace(0, 0.2, 1000).unsqueeze(1)
    norms = (norms > pert_sizes).float().mean(dim=1)

    ax = axes.flatten()[idx]
    ax.scatter(8/255, aa_acc[idx]/100, label='AA', marker='+', color='green', zorder=3)
    ax.text(8 / 255 + 0.01, aa_acc[idx]/100, f'{aa_acc[idx]/100:.3f}', fontsize=16, verticalalignment='center', color='#EE004D')

    #print(f"{norms[0]*100:.1f}")
    print(f"acc: {acc*100:.1f}")
    ax.plot(pert_sizes, norms, color='#3D5A80')
    ax.set_title(model.name.split('_')[2])
    ax.grid(True)

    custom_xticks = np.linspace(0, 0.2, 5)
    ax.set_xticks(custom_xticks)

    ax.axvline(x=8/255, color='#5DA271', linewidth=1, linestyle='--')
    ax.scatter(8/255, rob_acc, color='#EE6C4D', marker='*', label='FMN_b', zorder=3, s=30)
    ax.text(8/255 + 0.01, rob_acc+0.04, f'{rob_acc:.3f}', fontsize=16, verticalalignment='center', color='#EE6C4D')
    ax.legend()

figure.tight_layout(pad=2.0)
figure.text(0.5, 0.001, r'Perturbation $||\delta^*||$', ha='center', fontsize='large')
figure.text(0.001, 0.5, 'Robust Accuracy', va='center', rotation='vertical', fontsize='large')

plot_path = './Experiments/baseline_attacks/plots'
if not os.path.exists(plot_path):
    os.makedirs(plot_path, exist_ok=True)

plt.savefig(f"{plot_path}/fmn_attack_exps_1000samples_1000steps.pdf", bbox_inches='tight', dpi=320)
