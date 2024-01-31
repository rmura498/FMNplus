import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def extract_data(path, trial_list):
    folder = 'results/' + path
    filenames = os.listdir(folder)
    best_config_dict = {}

    for i, filename in enumerate(filenames):
        splitted = filename.split('_')
        name = f'{splitted[4]}-{splitted[5]}-{splitted[6]}'
        config_names = os.listdir(folder+filename)

        for config_name in config_names:
            conf = config_name.split('_')
            if conf[-1] == f'trial{trial_list[i]}.pth':
                f = torch.load(folder+filename+'/'+config_name)
                best_config_dict[name]=f
    return best_config_dict




def plot_robust():
    best_config_dict = extract_data(path='mid8_short_exps/', trial_list=[31,31,31,31,3])

    nrows = 2
    ncols = 3
    figure, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=True)

    for idx, conf in enumerate(best_config_dict.keys()):

        distance = best_config_dict[conf]['distance']
        best_distance = best_config_dict[conf]['best_distance']
        best_adv = best_config_dict[conf]['best_adv']
        inputs = best_config_dict[conf]['images']

        n_samples = best_adv.shape[0]
        norms = (best_adv - inputs).flatten(1).norm(torch.inf, dim=1)

        median_norms = norms.median().item()

        rob_acc = (norms > 8 / 255).float().mean()
        rob_acc_m = (norms > median_norms).float().mean()
        acc = (norms > 0).float().mean()

        pert_sizes = torch.linspace(0, 0.2, 1000).unsqueeze(1)
        norms = (norms > pert_sizes).float().mean(dim=1)


        ax = axes.flatten()[idx]
        # ax[i, j].scatter(8/255, aa_acc[models_dict[model]], label='AA', marker='+', color='green', zorder=3)
        # TODO: add AA point as the baseline/reference

        # print(f"{norms[0]*100:.1f}")
        print(f"acc: {acc * 100:.1f}")
        ax.plot(pert_sizes, norms, color='#3D5A80')
        ax.set_title(conf)
        ax.grid(True)

        custom_xticks = np.linspace(0, 0.2, 5)
        ax.set_xticks(custom_xticks)

        # closest_index = np.abs(pert_sizes - 8 / 255).argmin()
        # closest_value = pert_sizes[closest_index]
        # closest_norm = norms[closest_index]

        ax.axvline(x=median_norms, color='#5DA324', linewidth=1, linestyle='--')
        ax.scatter(median_norms, rob_acc_m, color='#EE6C4D', marker='*', label='median', zorder=3, s=30)
        ax.text(median_norms+0.01, rob_acc_m, f'{median_norms:.5f}', fontsize=12, verticalalignment='center', color='#EE6C4D')
        #ax.axvline(x=8 / 255, color='#5DA271', linewidth=1, linestyle='--')
        #ax.scatter(8 / 255, rob_acc, color='#EE6C4D', marker='*', label='8/255', zorder=3, s=30)
        #ax.text(8 / 255 + 0.01, rob_acc, f'{rob_acc:.3f}', fontsize=16, verticalalignment='center', color='#EE6C4D')
        ax.legend()

    figure.tight_layout(pad=1.5)
    figure.text(0.5, 0.01, r'Perturbation $||\delta^*||$', ha='center', fontsize='large')
    figure.text(0.001, 0.5, 'Robust Accuracy', va='center', rotation='vertical', fontsize='large')

    plt.savefig(f"comparison.pdf", bbox_inches='tight', dpi=320)


plot_robust()