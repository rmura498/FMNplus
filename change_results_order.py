import os
import pickle
import io

import torch
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
        model_folder = 'mid1'

    filenames = os.listdir('results' + f'/{model_folder}')

    fmn_config_dict = {}
    aa_config_dict = {}
    for filename in filenames:

        pkl_files = os.listdir(f'results/{model_folder}/{filename}')

        sr_list = []
        for pkl_file in pkl_files:

            with open(f'results/{model_folder}/{filename}/{pkl_file}', 'rb') as f:
                attack_data = CPU_Unpickler(f).load()
                sr = attack_data['success_rate']
                print(pkl_file, sr)
                sr_list.append(sr[-1])
        print(sr_list)
        mean_sr = sum(sr_list)/len(sr_list)
        print(mean_sr)
        new_filename = pkl_file.replace('_bid3','')
        attack_data = {
            'success_rate': mean_sr
        }
        with open(f'Experiments/201213_mid1/{new_filename}', 'wb') as f:
            pickle.dump(attack_data, f)

if __name__ == '__main__':
    main()
