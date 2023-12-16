import os
import pickle
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from config import EXP_DIRECTORY


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def main(model_folder=None):
    if model_folder is None:
        model_folder = '15122314_mid8'

    filenames = os.listdir(EXP_DIRECTORY + f'/{model_folder}')

    fmn_config_dict = {}
    aa_config_dict = {}
    for filename in filenames:
        with open(EXP_DIRECTORY + f'/{model_folder}/{filename}', 'rb') as f:
            attack_data = CPU_Unpickler(f).load()
            is_adv = attack_data['is_adv']
            sr = attack_data['success_rate']
        config_name = filename.split('_')
        config_key = f'{config_name[2]}-{config_name[6]}-{config_name[7]}-{config_name[8]}'
        batch_size = int(config_name[4])
        if config_name[10].replace('.pkl', '') == 'extraitersTrue':
            config_key = config_key + '-EI'
        if config_name[2] == 'AA':
            aa_config_dict[config_key] = [is_adv, sr]
        else:
            fmn_config_dict[config_key] = [is_adv, sr]

    fmn_sr_matrix = np.zeros((len(fmn_config_dict.keys()), len(fmn_config_dict.keys())), )

    for i, key_i in enumerate(fmn_config_dict):
        for j, key_j in enumerate(fmn_config_dict):
            base_is_adv = fmn_config_dict[key_i][0]
            base_sr = fmn_config_dict[key_i][1][-1]

            compared_is_adv = fmn_config_dict[key_j][0]
            compared_sr = fmn_config_dict[key_j][1][-1]

            if key_i == key_j:
                fmn_sr_matrix[i, j] = base_sr
            else:
                is_adv1_index = (base_is_adv == True).float().nonzero()
                is_adv2_index = (compared_is_adv == True).float().nonzero()
                all_adv_index = np.concatenate((is_adv1_index, is_adv2_index), axis=None)
                unique_adv_index = np.unique(all_adv_index)
                fmn_sr_matrix[i, j] = len(unique_adv_index) / batch_size * 100

    config_list = list(fmn_config_dict.keys())
    divide_plots = int(len(fmn_config_dict.keys()) / 2)
    df_cm1 = pd.DataFrame(fmn_sr_matrix[:divide_plots, :divide_plots],
                          index=[i for i in config_list[:divide_plots]],
                          columns=[i for i in config_list[:divide_plots]])

    df_cm2 = pd.DataFrame(fmn_sr_matrix[divide_plots:, divide_plots:],
                          index=[i for i in config_list[divide_plots:]],
                          columns=[i for i in config_list[divide_plots:]])
    fig, ax = plt.subplots(1, 2, figsize=(42, 30))
    ax = ax.flatten()
    sn.heatmap(df_cm1, ax=ax[0], annot=True, fmt='.2f', cmap='crest')
    sn.heatmap(df_cm2, ax=ax[1], annot=True, fmt='.2f', cmap='crest')
    plt.savefig(f'FMN-{model_folder}.pdf')

    aa_sr_matrix = np.zeros((len(aa_config_dict.keys()), len(aa_config_dict.keys())), )

    for i, key_i in enumerate(aa_config_dict):
        for j, key_j in enumerate(aa_config_dict):
            base_is_adv = aa_config_dict[key_i][0]
            base_sr = aa_config_dict[key_i][1][-1]

            compared_is_adv = aa_config_dict[key_j][0]
            compared_sr = aa_config_dict[key_j][1][-1]

            if key_i == key_j:
                aa_sr_matrix[i, j] = base_sr.numpy()
            else:
                is_adv1_index = (base_is_adv == True).float().nonzero()
                is_adv2_index = (compared_is_adv == True).float().nonzero()
                all_adv_index = np.concatenate((is_adv1_index, is_adv2_index), axis=None)
                unique_adv_index = np.unique(all_adv_index)
                aa_sr_matrix[i, j] = float("{:.3f}".format(len(unique_adv_index) / batch_size * 100))

    df_cm = pd.DataFrame(aa_sr_matrix, index=[i for i in aa_config_dict.keys()],
                         columns=[i for i in aa_config_dict.keys()])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='.1f')
    plt.savefig(f'AA-{model_folder}.pdf')


if __name__ == '__main__':
    main(model_folder='15122314_mid8')
