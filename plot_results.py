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


def read_results(filenames=[], attack_type='FMNBase', sign=-1):
    print("Read Results", attack_type)
    if len(filenames) == 0:
        print("Invalid filenames")
        return None

    loss_list = []
    sr_list = []

    for filename in filenames:
        with open(filename, 'rb') as f:
            attack_data = CPU_Unpickler(f).load()
            loss = attack_data['loss']
            success_rate = attack_data['success_rate']
            if attack_type == 'AA':
                loss_list.append(sign * loss)
            else:
                loss_list.append(loss)
            sr_list.append(success_rate)

    if attack_type == 'AA':
        sr_list = [sr for sr in sr_list[-1]]
        sr_batch = torch.stack(sr_list, dim=0)
        #mean_sr = torch.mean(sr_batch, dim=0)
        mean_sr = sr_batch
        batch_loss = torch.stack(loss_list, dim=0)
        #mean_loss = torch.mean(batch_loss, dim=0)
        mean_loss = batch_loss.reshape(100)
    else:
        loss = torch.tensor(loss_list[-1])
        sr = torch.tensor(sr_list[-1])
        mean_loss = loss
        mean_sr = sr
        if len(loss) == 300:
            loss = loss.reshape(3, 100)
            sr = sr.reshape(3, 100)
            mean_loss = torch.mean(loss, dim=0)
            mean_sr = torch.mean(sr, dim=0)

    return mean_sr, mean_loss


def plot_model_results(model_id='1'):
    attack_exps = os.listdir(f'Exps/13122314')

    filenames_DLR = [f'Exps/13122314/' + filename for filename in attack_exps if
                     filename.split('-')[5] == 'lossDLR']
    filenames_CE = [f'Exps/13122314/' + filename for filename in attack_exps if
                    filename.split('-')[5] == 'lossCE']
    filenames_LL = [f'Exps/13122314/' + filename for filename in attack_exps if
                    filename.split('-')[5] == 'lossLL']

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 8))

    fig.suptitle(f'Model{model_id}')
    x = np.linspace(0, 100, 100)
    for filename in filenames_DLR:
        pickle_files = os.listdir(filename)
        attack_type = filename.split('-')[0].split('/')[-1]
        print(attack_type)
        path = filename + '/'

        if attack_type != 'AA':
            optAdam = [exp for exp in pickle_files if exp.split('_')[1] == 'Adam']
            optSGD = [exp for exp in pickle_files if exp.split('_')[1] == 'SGD']
            if optAdam == [] or optSGD == []:
                continue

            #mean_sr, mean_loss = read_results([path + file for file in optAdam], attack_type)
            for filename in optAdam:
                mean_sr, mean_loss = read_results([path + filename,], attack_type)
                loss = mean_loss.numpy()
                sr = mean_sr.numpy()
                filename = filename.split('_')
                ax.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[-1])}-optAdam-sch{filename[2]}')

            for filename in optSGD:
                mean_sr, mean_loss = read_results([path + filename,], attack_type)
                loss = mean_loss.numpy()
                sr = mean_sr.numpy()
                filename = filename.split('_')
                ax.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[-1])}-optSGD-sch{filename[2]}')
        else:
            mean_sr, mean_loss = read_results([path + file for file in pickle_files], attack_type)
            loss = mean_loss.numpy()
            sr = mean_sr.numpy()
            ax.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[0])}')

    for filename in filenames_CE:
        pickle_files = os.listdir(filename)
        attack_type = filename.split('-')[0].split('/')[-1]
        print(attack_type)
        path = filename + '/'

        if attack_type != 'AA':
            optAdam = [exp for exp in pickle_files if exp.split('_')[1] == 'Adam']

            optSGD = [exp for exp in pickle_files if exp.split('_')[1] == 'SGD']
            print(optAdam)
            if optAdam == [] or optSGD == []:
                continue

            for filename in optAdam:
                mean_sr, mean_loss = read_results([path + filename,], attack_type)
                loss = mean_loss.numpy()
                sr = mean_sr.numpy()
                filename = filename.split('_')
                ax1.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[-1])}-optAdam-sch{filename[2]}')

            for filename in optSGD:
                mean_sr, mean_loss = read_results([path + filename,], attack_type)
                loss = mean_loss.numpy()
                sr = mean_sr.numpy()
                filename = filename.split('_')
                ax1.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[-1])}-optSGD-sch{filename[2]}')

        else:
            mean_sr, mean_loss = read_results([path + file for file in pickle_files], attack_type)
            loss = mean_loss.numpy()
            sr = mean_sr.numpy()
            ax1.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[0])}')

    for filename in filenames_LL:
        pickle_files = os.listdir(filename)
        attack_type = filename.split('-')[0].split('/')[-1]
        print(attack_type)
        path = filename + '/'

        if attack_type != 'AA':
            optAdam = [exp for exp in pickle_files if exp.split('_')[1] == 'Adam']
            optSGD = [exp for exp in pickle_files if exp.split('_')[1] == 'SGD']
            if optAdam == [] or optSGD == []:
                continue

            for filename in optAdam:
                mean_sr, mean_loss = read_results([path + filename,], attack_type)
                loss = mean_loss.numpy()
                sr = mean_sr.numpy()
                filename = filename.split('_')
                ax2.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[-1])}-optAdam-sch{filename[2]}')

            for filename in optSGD:
                mean_sr, mean_loss = read_results([path + filename,], attack_type)
                loss = mean_loss.numpy()
                sr = mean_sr.numpy()
                filename = filename.split('_')
                ax2.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[-1])}-optSGD-sch{filename[2]}')

        else:
            mean_sr, mean_loss = read_results([path + file for file in pickle_files], attack_type)
            loss = mean_loss.numpy()
            sr = mean_sr.numpy()
            ax2.plot(x, loss, label=f'{attack_type}-SR:{"{:.3f}".format(sr[0])}')

    ax.set_xlabel('Steps')
    ax.set_ylabel('DLR Losses')
    ax.legend()
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('CE Losses')
    ax1.legend()
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('LL Losses')
    ax2.legend()

    plt.savefig('output.pdf')

    plt.show()


plot_model_results(model_id='8')
