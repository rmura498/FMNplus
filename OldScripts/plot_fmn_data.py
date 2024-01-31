# TODO: modify this for plotting the trials results

import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import argparse

# plt.style.use(['science','ieee'])

parser = argparse.ArgumentParser()
parser.add_argument('--fmn_source', choices=["hydra", "harp"], default='hydra')
args = parser.parse_args()

data_dir = pathlib.Path(f'{args.fmn_source}_fmn_attack_data')
models = tuple({'_'.join(f.name.split('_')[:2])
                for f in data_dir.glob('*') if f.is_dir()})
models = sorted(models)

# change this values
nrows = 1
ncols = 4
figure, axes = plt.subplots(nrows, ncols, figsize=(12, 3), sharex=True, sharey=True)

for idx, model in enumerate(models):
    model_dir = list(data_dir.glob(f'{model}*'))[0]
    batches = [f for f in model_dir.glob('*')]

    best_advs = None
    inputs = None
    for batch in batches[:len(batches)]:
        _best_advs = torch.load(batch / 'best_adv.pt', map_location='cpu')
        _inputs = torch.load(batch / 'inputs.pt', map_location='cpu')

        best_advs = _best_advs if best_advs is None else torch.cat([best_advs, _best_advs])
        inputs = _inputs if inputs is None else torch.cat([inputs, _inputs])

    n_samples = best_advs.shape[0]
    norms = (best_advs - inputs).flatten(1).norm(torch.inf, dim=1)

    rob_acc = (norms>8/255).float().mean()
    acc = (norms>0).float().mean()

    pert_sizes = torch.linspace(0, 0.2, 1000).unsqueeze(1)
    norms = (norms > pert_sizes).float().mean(dim=1)

    ax = axes.flatten()[idx]
    # ax[i, j].scatter(8/255, aa_acc[models_dict[model]], label='AA', marker='+', color='green', zorder=3)
    # TODO: add AA point as the baseline/reference

    #print(f"{norms[0]*100:.1f}")
    print(f"acc: {acc*100:.1f}")
    ax.plot(pert_sizes, norms, color='#3D5A80')
    ax.set_title(model)
    ax.grid(True)

    custom_xticks = np.linspace(0, 0.2, 5)
    ax.set_xticks(custom_xticks)

    #closest_index = np.abs(pert_sizes - 8 / 255).argmin()
    #closest_value = pert_sizes[closest_index]
    #closest_norm = norms[closest_index]

    ax.axvline(x=8/255, color='#5DA271', linewidth=1, linestyle='--')
    ax.scatter(8/255, rob_acc, color='#EE6C4D', marker='*', label='8/255', zorder=3, s=30)
    ax.text(8/255 + 0.01, rob_acc, f'{rob_acc:.3f}', fontsize=16, verticalalignment='center', color='#EE6C4D')
    ax.legend()

figure.tight_layout(pad=1.5)
figure.text(0.5, 0.01, r'Perturbation $||\delta^*||$', ha='center', fontsize='large')
figure.text(0.001, 0.5, 'Robust Accuracy', va='center', rotation='vertical', fontsize='large')

plt.savefig(f"{args.fmn_source}_fmn_attack_exps_{n_samples}samples_500steps.pdf", bbox_inches='tight', dpi=320)
