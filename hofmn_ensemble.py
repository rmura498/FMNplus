import os, pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=15)
plt.rc('font', size=13)

aa_acc = [70.69, 67.31, 66.10, 64.58, 63.38, 63.35, 62.79, 61.04, 58.50]

# load apgd robust values
apgd_dir = pathlib.Path('Experiments/apgd')
apgd_attacks = tuple({f for f in apgd_dir.glob('*') if (f.is_dir() and 'mid' in f.name)})
apgd_attacks = sorted(apgd_attacks)
apgd_robust_values = dict()

for apgd_attack in apgd_attacks:
    model_name_split = list(apgd_attack.name.split('_'))
    _model_id = int(model_name_split[2].replace('mid',''))
    _loss_fn = str(model_name_split[-1])

    rob_value = None

    batches = [f for f in apgd_attack.glob('*')]
    robust_values = []
    for batch in batches[:len(batches)]:
        data = torch.load(batch, map_location='cpu')
        robust_values.append(data['ra'])

    rob_value = torch.tensor(robust_values, device='cpu').mean().item()

    if _model_id not in apgd_robust_values:
        apgd_robust_values[_model_id] = dict()
    apgd_robust_values[_model_id][_loss_fn] = rob_value


data_dir = pathlib.Path(f'./Experiments/best_attacks')
baseline_dir = pathlib.Path(f'./Experiments/baseline_attacks')

models = tuple({f for f in data_dir.glob('*') if (f.is_dir() and 'mid' in f.name)})
models = sorted(models)

baseline_models = tuple({f for f in baseline_dir.glob('*') if (f.is_dir() and 'mid' in f.name)})
baseline_models = sorted(baseline_models)

print(f"Baseline models: {baseline_models}")
print(f"Best models: {models}")

# change this values
nrows = 3
ncols = 3
figure, axes = plt.subplots(nrows, ncols, figsize=(15,11), dpi=320, sharex=True, sharey=True)

pert_sizes = torch.linspace(0, 0.15, 1000).unsqueeze(1)

baseline_robust_values = dict()
baseline_norms = list()
for idx, baseline_model in enumerate(baseline_models):
    model_id = int(baseline_model.name.split('_')[2].replace('mid',''))
    # print(f"Baseline: idx {idx} == {model_id}")

    batches = [f for f in baseline_model.glob('*')]
    best_dist = []
    best_advs = None
    inputs = None
    for batch in batches[:len(batches)]:
        attack_data = torch.load(batch, map_location='cpu')

        _best_advs = attack_data['best_adv']
        _inputs = attack_data['images']

        best_advs = _best_advs if best_advs is None else torch.cat([best_advs, _best_advs])
        inputs = _inputs if inputs is None else torch.cat([inputs, _inputs])
        best_dist.append(torch.linalg.norm((_best_advs - _inputs), ord=torch.inf, dim=1).median().item())
    best_dist_base = torch.tensor(best_dist).mean().item()
    n_samples = best_advs.shape[0]
    norms_base = (best_advs - inputs).flatten(1).norm(torch.inf, dim=1)
    _norms_base_clone = norms_base.clone()
    baseline_norms.append(_norms_base_clone)

    _rob_acc = (norms_base > 8 / 255).float().mean()
    baseline_robust_values[model_id] = _rob_acc.item()

    norms_base = (norms_base > pert_sizes).float().mean(dim=1)

    # plot baseline norms
    ax = axes.flatten()[idx]
    ax.plot(pert_sizes, norms_base, color='grey', label='FMN', linestyle='--')

    ax.set_title(baseline_model.name.split('_')[2])
    ax.grid(True)

    custom_xticks = np.linspace(0, 0.2, 5)
    ax.set_xticks(custom_xticks)

    custom_yticks = np.linspace(0, 1.0, 5)
    ax.set_yticks(custom_yticks)


ensamble_rob_path = os.path.join('Experiments', 'hofmn_ensamble.pt')
if os.path.exists(ensamble_rob_path):
    ensamble_rob = torch.load(ensamble_rob_path)
else:
    ensamble_rob = dict()
    ensamble_norms = torch.full((3, 1000), fill_value=torch.inf, device='cpu')

    j = 0
    for model in models:
        idx = int(model.name.split('_')[2].replace('mid', ''))
        loss_fn = str(list(model.name.split('_'))[-3])

        conf_name = f"{model.name.split('_')[5]}-{model.name.split('_')[6]}-{model.name.split('_')[7]}"

        print(f"Model id: {idx} - conf: {conf_name}")

        batches = [f for f in model.glob('*')]
        # best_dist = []
        best_advs = None
        inputs = None
        for batch in batches[:len(batches)]:
            attack_data = torch.load(batch, map_location='cpu')

            _best_advs = attack_data['best_adv']
            _inputs = attack_data['images']

            best_advs = _best_advs if best_advs is None else torch.cat([best_advs, _best_advs])
            inputs = _inputs if inputs is None else torch.cat([inputs, _inputs])
            # best_dist.append(torch.linalg.norm((_best_advs-_inputs), ord=torch.inf, dim=1).median().item())
        # best_dist = torch.tensor(best_dist).mean().item()
        n_samples = best_advs.shape[0]
        print(f"Samples: {n_samples}, j: {j}")
        norms = (best_advs - inputs).flatten(1).norm(torch.inf, dim=1)

        ensamble_norms[j][:n_samples] = norms
        j += 1

        if j == 3:
            ensamble_rob[idx] = ensamble_norms
            ensamble_norms = torch.full_like(ensamble_norms, fill_value=torch.inf, device='cpu')
            j = 0

    print(f"Shape of first norms: {ensamble_rob[0].shape}")

    for model in ensamble_rob:
        ensamble_rob[model] = torch.min(ensamble_rob[model], dim=0).values

    torch.save(ensamble_rob, os.path.join('Experiments', 'hofmn_ensamble.pt'))

robust_df = []
for idx in ensamble_rob:
    norms = ensamble_rob[idx]
    _norms_clone = norms.clone()
    rob_acc = (norms > 8 / 255).float().mean()

    acc = (norms > 0).float().mean().item()
    rob_acc = (norms > 8/255).float().mean().item()
    norms = (norms > pert_sizes).float().mean(dim=1)

    # check if we beat apgd
    apgd_rob_sorted = sorted(list(apgd_robust_values[idx].values()))
    apgd_best_point = apgd_rob_sorted[0]
    print(f"M{idx+1}...")
    print(f"HO-FMN rob: {rob_acc}")
    print(f"APGD Best rob: {apgd_best_point}")

    robust_df.append([f'M{idx+1}', rob_acc, apgd_best_point, aa_acc[idx]])

    ax = axes.flatten()[idx]
    ax.plot(pert_sizes, norms, color='blue', label='HO-FMN (ens.)', lw=2)
    ax.set_title(f'$M_{{{idx + 1}}}$ top-3 ens.')
    # ax.grid(True)

    ax.axvline(x=8 / 255, color='#EE004D', linewidth=1, linestyle='--')
    ax.scatter(8 / 255, apgd_best_point, label='APGD', marker='+', color='#EE004D', zorder=3, s=10 ** 2)

    ax.set_xlim(0.0, 0.15)
    ax.set_ylim(0.0, 1.0)

    # inset axes....
    x1, x2, y1, y2 = 7 / 255, 9 / 255, rob_acc - (rob_acc / 14), rob_acc + (
                rob_acc / 12)  # subregion of the original image

    zoomed_x = torch.linspace(x1, x2, 200).unsqueeze(1)
    zoomed_y = (_norms_clone > zoomed_x).float().mean(dim=1)
    zoomed_y_base = (baseline_norms[idx] > zoomed_x).float().mean(dim=1)

    axins = ax.inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

    axins.plot(zoomed_x, zoomed_y, color='blue', lw=1.5)
    axins.plot(zoomed_x, zoomed_y_base, color='grey', linestyle='--')
    axins.scatter(8 / 255, apgd_best_point, label='APGD', marker='+', color='#EE004D', zorder=3, s=10 ** 2)

    ax.indicate_inset_zoom(axins, edgecolor="black")

figure.tight_layout(pad=1.5)
# figure.text(0.5, 0.001, 'Perturbation Budget', ha='center', fontsize='large')
# figure.text(0.001, 0.5, 'Robust Accuracy', va='center', rotation='vertical', fontsize='large')

plot_path = './Experiments/best_attacks/plots'
if not os.path.exists(plot_path):
    os.makedirs(plot_path, exist_ok=True)

handles, labels = axes.flatten()[0].get_legend_handles_labels()
axes.flatten()[7].legend(handles, labels, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5,-0.3), fancybox=True, shadow=True)
# figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5,0.0), ncol=len(labels))

plt.savefig(f"{plot_path}/ho_fmn_ensemble.pdf", bbox_inches='tight', dpi=320)

# save robust df
robust_df = pd.DataFrame(robust_df, columns=['Model', 'HO-FMN', 'APGD', 'AA'])
print(robust_df)
robust_df.to_csv(f"{plot_path}/ho_fmn_ensemble_robust.csv")