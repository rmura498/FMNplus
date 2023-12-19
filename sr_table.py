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
            sr = attack_data['success_rate']
        config_name = filename.split('_')
        config_key = f'{config_name[2]}-{config_name[6]}-{config_name[7]}-{config_name[8]}'
        batch_size = int(config_name[4])
        if config_name[10].replace('.pkl', '') == 'extraitersTrue':
            config_key = config_key + '-EI'

        if config_name[2] == 'AA':
            aa_config_dict[config_key] = sr[-1]
        else:
            fmn_config_dict[config_key] = sr[-1]

    fmn_sr_series = pd.Series(fmn_config_dict.values(), index=fmn_config_dict.keys())

    return fmn_sr_series

if __name__ == '__main__':
    exp_list = os.listdir(EXP_DIRECTORY)

    fmn_sr_dict = {}
    for model in exp_list:
        fmn_sr_dict[f"SR-{model.split('_')[-1]}"] = main(model_folder=model)

    fmn_sr_df = pd.DataFrame(data=fmn_sr_dict, index=fmn_sr_dict[list(fmn_sr_dict.keys())[0]].index)
    fmn_sr_df['SR-mean'] = fmn_sr_df.mean(axis=1)
    fmn_sr_df = fmn_sr_df.sort_values('SR-mean', ascending=False)

    fmn_sr_df_styled = fmn_sr_df.style.highlight_max(props='cellcolor:[HTML]{b666cc}; color:{white};' 'textit:--rwrap; textbf:--rwrap;').format("{:.2f}")

    fmn_sr_df_styled.to_latex('fmn_sr_comparison_181223.txt')

