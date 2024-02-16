import os, pathlib

import torch
import numpy as np
import matplotlib.pyplot as plt

import threading

plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=15)
plt.rc('font', size=13)

'''
0
[('Adamax-None-DLR', 0.7179999947547913), ('SGD-CALR-DLR', 0.7170000076293945)]
1
[('Adam-None-DLR', 0.6834285855293274)]
2
[('Adam-None-DLR', 0.6830000281333923), ('Adamax-None-DLR', 0.6809999942779541), ('SGD-CALR-DLR', 0.6779999732971191)]
3
[('Adam-None-DLR', 0.6610000133514404), ('SGD-CALR-DLR', 0.6570000052452087), ('SGD-CALR-LL', 0.6610000133514404)]
4
[('Adam-None-DLR', 0.6520000100135803), ('SGD-CALR-DLR', 0.6520000100135803)]
6
[('Adam-None-DLR', 0.6380000114440918), ('SGD-CALR-DLR', 0.6380000114440918), ('SGD-CALR-LL', 0.640999972820282)]
8
[('SGD-CALR-DLR', 0.5960000157356262), ('SGD-CALR-LL', 0.609000027179718)]
'''

top_conf = {
    0: 'SGD-CALR-DLR',
    1: 'Adam-None-DLR',
    2: 'SGD-CALR-DLR',
    3: 'SGD-CALR-DLR',
    4: 'SGD-CALR-DLR',
    5: 'SGD-CALR-DLR',
    6: 'SGD-CALR-DLR',
    7: 'SGD-CALR-DLR',
    8: 'SGD-CALR-DLR'
}

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

j = 0
# best_robs = []
# apgd_best_robs = []

win = 0
winner_confs = dict()

j_plot_top = 0

hofmn_top_robust_values = dict()
for model in models:
    idx = int(model.name.split('_')[2].replace('mid', ''))
    loss_fn = str(list(model.name.split('_'))[-3])

    conf_name = f"{model.name.split('_')[5]}-{model.name.split('_')[6]}-{model.name.split('_')[7]}"

    # do not plot if not top-1
    if conf_name not in top_conf[idx]:
        continue

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
    _norms_clone = norms.clone()

    rob_acc = (norms>8/255).float().mean().item()
    acc = (norms>0).float().mean().item()

    norms = (norms > pert_sizes).float().mean(dim=1)

    if idx not in hofmn_top_robust_values:
        hofmn_top_robust_values[idx] = dict()
    hofmn_top_robust_values[idx][conf_name] = rob_acc.item()

    # check if we beat apgd
    apgd_rob_sorted = sorted(list(apgd_robust_values[idx].values()))
    apgd_best_point = apgd_rob_sorted[0]
    # apgd_best_robs.append(apgd_best_point)
    print(f"Conf name: {conf_name}")
    print(f"HO-FMN rob: {rob_acc}")
    print(f"APGD Best rob: {apgd_best_point}")

    ax = axes.flatten()[j_plot_top]
    # ax.scatter(8/255, aa_acc[idx]/100, label='AA', marker='+', color='green', zorder=3)
    # ax.text(8 / 255 + 0.01, aa_acc[idx]/100, f'{aa_acc[idx]/100:.3f}', fontsize=16, verticalalignment='center', color='#EE004D')

    print(f"acc: {acc*100:.1f}")
    ax.plot(pert_sizes, norms, color='blue', label='HO-FMN', lw=2)
    if 'SGD' in conf_name:
        conf_name = conf_name.replace('SGD', 'GD')
    ax.set_title(f'$M_{{{idx+1}}}$: {conf_name}')
    #ax.grid(True)

    ax.axvline(x=8/255, color='#EE004D', linewidth=1, linestyle='--')
    # ax.scatter(8/255, rob_acc, color='#EE6C4D', marker='*', label='FMN_best', zorder=3, s=30)
    # ax.text(8/255 + 0.01, rob_acc+0.05, f'{rob_acc:.3f}', fontsize=16, verticalalignment='center', color='#EE6C4D')

    # AA point
    # ax.scatter(8 / 255, aa_acc[idx] / 100, label='AA', marker='+', color='#EE004D', zorder=3)
    # ax.text(8 / 255 + 0.01, aa_acc[idx] / 100, f'{aa_acc[idx] / 100:.3f}', fontsize=16, verticalalignment='center',
    #        color='#EE004D')

    # apgd point
    '''
    if idx in apgd_robust_values and loss_fn != 'LL':
        apgd_rob = apgd_robust_values[idx][loss_fn]
        ax.scatter(8 / 255, apgd_rob, label='apgd', marker='+', color='#EE004D', zorder=3)
        ax.text(8 / 255 + 0.01, apgd_rob + 0.05, f'{apgd_rob:.3f}', fontsize=16, verticalalignment='center',
                color='#EE004D')
    else:
        ax.scatter(8 / 255, aa_acc[idx] / 100, label='AA', marker='+', color='#EE004D', zorder=3)
        ax.text(8 / 255 + 0.01, aa_acc[idx] / 100, f'{aa_acc[idx] / 100:.3f}', fontsize=16, verticalalignment='center',
                color='#EE004D')
    '''
    if rob_acc <= apgd_best_point:
        win += 1
        if idx not in winner_confs:
            winner_confs[idx] = list()
        winner_confs[idx].append((conf_name, rob_acc.item()))

    ax.scatter(8 / 255, apgd_best_point, label='APGD', marker='+', color='#EE004D', zorder=3, s=10**2)
    # ax.text(8 / 255 + 0.01, apgd_best_point + 0.05, f'{apgd_best_point:.3f}', fontsize=16, verticalalignment='center',
    #         color='#EE004D')

    # ax.legend()

    ax.set_xlim(0.0, 0.15)
    ax.set_ylim(0.0, 1.0)

    # inset axes....
    x1, x2, y1, y2 = 7/255, 9/255, rob_acc-(rob_acc/14), rob_acc+(rob_acc/12)   # subregion of the original image

    zoomed_x = torch.linspace(x1, x2, 200).unsqueeze(1)
    zoomed_y = (_norms_clone > zoomed_x).float().mean(dim=1)
    zoomed_y_base = (baseline_norms[idx] > zoomed_x).float().mean(dim=1)

    axins = ax.inset_axes(
        [0.5, 0.5, 0.47, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])

    axins.plot(zoomed_x, zoomed_y, color='blue', lw=1.5)
    axins.plot(zoomed_x, zoomed_y_base, color='grey', linestyle='--')
    axins.scatter(8 / 255, apgd_best_point, label='APGD', marker='+', color='#EE004D', zorder=3, s=10**2)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    j_plot_top += 1

print(f"How many times HO-FMN beated apgd: {win}")
print(f"Winner confs: ")
for w_idx in winner_confs:
    print(w_idx)
    print(winner_confs[w_idx])

# torch.save({'best_robs': best_robs, 'apgd_best_robs': apgd_best_robs}, 'best_vs_apgd_rob_acc.pt')

figure.tight_layout(pad=1.5)
# figure.text(0.5, 0.001, 'Perturbation Budget', ha='center', fontsize='large')
# figure.text(0.001, 0.5, 'Robust Accuracy', va='center', rotation='vertical', fontsize='large')

plot_path = './Experiments/best_attacks/plots'
if not os.path.exists(plot_path):
    os.makedirs(plot_path, exist_ok=True)


handles, labels = axes.flatten()[0].get_legend_handles_labels()
axes.flatten()[7].legend(handles, labels, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5,-0.3), fancybox=True, shadow=True)
# figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5,0.0), ncol=len(labels))

plt.savefig(f"{plot_path}/fmn_best_attack_exps_1000samples_1000steps.pdf", bbox_inches='tight', dpi=320)

# saving data
torch.save({
    'apgd': apgd_robust_values,
    'baseline': baseline_robust_values,
    'hofmn_top': hofmn_top_robust_values
}, 'robust_values.pt')