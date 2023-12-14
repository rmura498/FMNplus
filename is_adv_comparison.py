"""
Funzione che prende in ingresso il model id e loss da considerare

chiama read_results che carica i pkl e ne fa la media -> resistiuisce gli array da plottare
ovvero 5 array di loss (5 curve AA/2 FMNBase / 2 FMNVec)

chiami il plot

"""

import pickle
import numpy as np
import torch
import os
import io
import matplotlib.pyplot as plt

device = torch.device("cpu")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def read_results(filename):
    print("Read Results")
    if len(filename) == 0:
        print("Invalid filenames")
        return None




    with open(filename, 'rb') as f:
        attack_data = CPU_Unpickler(f).load()
        is_adv = attack_data['is_adv']


    return is_adv



def is_adv_comparison(filename1, filename2,filename3=None, batch_size=500):
    is_adv1 = read_results(filename1)
    is_adv2 = read_results(filename2)

    is_adv1_index = (is_adv1 == True).float().nonzero()
    is_adv2_index = (is_adv2 == True).float().nonzero()

    #is_adv1_index = set(is_adv1_index.numpy())
    #is_adv2_index = set(is_adv2_index.numpy())
    #comparison = is_adv1_index.intersection(is_adv2_index)
    if filename3 == None:
        comparison = np.intersect1d(is_adv2_index, is_adv1_index)
        print(f'IS ADV 1{is_adv1_index.shape}')
        print(f'IS ADV 2{is_adv2_index.shape}')
        total_adv = (is_adv2_index.shape[0]+is_adv1_index.shape[0]) - len(comparison)
        print('equal', len(comparison))
        print(f'COMP {len(comparison)/(is_adv2_index.shape[0]+is_adv1_index.shape[0])*100}', '%')
        print('total adv ', total_adv)
        print(f'SR: {total_adv/batch_size*100}', '%')

    else:
        is_adv3 = read_results(filename3)
        is_adv3_index = (is_adv3 == True).float().nonzero()
        comparison = np.intersect1d(is_adv2_index, is_adv1_index)
        comparison12 = np.intersect1d(is_adv2_index, is_adv1_index)
        comparison13 = np.intersect1d(is_adv1_index, is_adv3_index)
        comparison23 = np.intersect1d(is_adv2_index, is_adv3_index)
        print(f'IS ADV 1{is_adv1_index.shape}')
        print(f'IS ADV 2{is_adv2_index.shape}')
        print(f'IS ADV 3{is_adv3_index.shape}')
        total_comparison = np.intersect1d(comparison, is_adv3_index)
        print('common', (len(comparison13)+len(comparison12)+len(comparison23)))
        total_adv = (is_adv2_index.shape[0] + is_adv1_index.shape[0]+is_adv3_index.shape[0]) - (len(comparison13)+len(comparison12)+len(comparison23))
        print('equal', len(total_comparison))
        print(f'COMP {len(total_comparison) / (is_adv2_index.shape[0] + is_adv1_index.shape[0]+is_adv3_index.shape[0]) * 100}', '%')
        print('total adv ', total_adv)
        print(f'SR: {total_adv / batch_size * 100}', '%')



is_adv_comparison('Exps/13122314/FMNBase-eps8-255-bs500-steps100-lossCE--gradientSign/13122314_SGD_CALR_steps100_batch500.pkl',
                  'Exps/13122314/FMNBase-eps8-255-bs500-steps100-lossDLR--gradientSign/13122314_SGD_CALR_steps100_batch500.pkl',
                  'Exps/13122314/FMNBase-eps8-255-bs500-steps100-lossLL--gradientSign/13122314_Adam_RLROP_steps100_batch500.pkl')
