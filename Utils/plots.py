import torch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Utils.compute_robust import compute_robust
# from Utils.metrics import compute_robust


def NormalizeData(data):
    return 2*((data - np.min(data)) / (np.max(data) - np.min(data))) -1
    #return ((data - np.min(data)) / (np.max(data) - np.min(data)))


def plot_distance():
    filenames = os.listdir('./experiments/')

    fig, axs = plt.subplots(8, 6, figsize=(25,25), layout='constrained')

    for ax, filename in zip(axs.flat, filenames):
        with open('./experiments/' + filename, 'rb') as file:
            attack_data = pickle.load(file)

        distance = attack_data['distance']
        distance_sample = [distance[0].tolist() for distance in distance]
        epsilon = attack_data['epsilon']
        epsilon_sample = [epsilon[0].tolist() for epsilon in epsilon]
        loss_attack = attack_data['loss']
        loss_sample = [loss[0].tolist() for loss in loss_attack]

        min_dist = "{:.7f}".format(distance_sample[-1])
        #epsilons = NormalizeData(epsilon_sample)
        distances = NormalizeData(distance_sample)
        #loss = NormalizeData(loss_sample)
        steps = len(epsilon_sample)

        #ax.plot(epsilons, label='epsilon')
        #ax.plot(distances, label='distances')
        ax.plot(loss_sample, label='loss')
        ax.legend(loc=0, prop={'size': 4})

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4

        title = filename.split('/')
        title = title[-1].split('.')
        ax.set_title(f"Steps: {steps}, S: {title[0]}, MinDist: {min_dist}", fontsize=fontsize)

        ax.set_xlabel("Steps")
        ax.set_ylabel("Distance - Loss")
        ax.grid()

    plt.show()
    fig.savefig("example.pdf")


def plot_epsilon_robust(distance,
                        best_distances, aa_robust=1):
    if len(distance) == 0:
        print("Error: Not enough distances per experiment!")
        return

    fig, ax = plt.subplots()
    # single experiment
    steps = len(distance)

    distances, robust_per_iter = compute_robust(distance, best_distances)

    distances = np.array(distances)
    distances.sort()
    robust_per_iter.sort(reverse=True)

    ax.plot(distances,
            robust_per_iter)
    #ax.plot(8/255, aa_robust, 'x', label='AA')
    ax.grid()

    dpi = fig.dpi
    rect_height_inch = ax.bbox.height / dpi
    fontsize = rect_height_inch * 4
    ax.set_title(f"Steps: {steps}",
                 fontsize=fontsize)
    ax.legend(loc=0, prop={'size': 8})
    ax.set_xlabel(r"$||\boldsymbol{\delta}||_\infty$")
    ax.set_ylabel("R. Acc.")


    plt.xlim([0, 0.2])
    fig.savefig("example.pdf")


def list_files_in_folders(directory):
    result_dict = {}

    # Iterate over each folder in the specified directory
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        # Check if the item in the directory is a folder
        if os.path.isdir(folder_path):
            # Retrieve the list of files in the folder
            files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            # Create a dictionary entry with folder_name as key and files_in_folder as value
            result_dict[folder_name] = files_in_folder

    return result_dict


def plot_loss_AA_fmn_comparison(directory='../experiments/loss_AA_FMN_comparison/Gowal2021Improving_R18_ddpm_100m/cifar10'):
    # Call the function to retrieve the desired dictionary
    result_dictionary = list_files_in_folders(directory)

    # Print the result
    for folder_name, files_in_folder in result_dictionary.items():
        print(f"{folder_name}: {files_in_folder}")

    for i in range(4):
        fig, axs = plt.subplots(8, 6, figsize=(25, 25), layout='constrained')
        axs_flat = axs.flatten()

        for j, (folder_name, files_in_folder) in enumerate(result_dictionary.items()):
            with open(os.path.join(directory, folder_name, files_in_folder[i]), 'rb') as file:
                loss_strat = pickle.load(file)

            axs_flat[j].plot(loss_strat['Loss_AA'], label='loss_AA')
            axs_flat[j].plot(loss_strat['Loss_fmn'], label='loss_fmn')
            axs_flat[j].legend()

            title = files_in_folder[i].split('/')
            title = title[-1].split('.')
            axs_flat[j].set_title(f"sample{folder_name}")
            axs_flat[j].set_xlabel("Steps")
            axs_flat[j].set_ylabel("Loss")
            axs_flat[j].grid()
        fig.savefig(f"{files_in_folder[i].replace('.pkl', '')}.pdf")

'''
if plot:
    # plot losses
    fig, ax = plt.subplots(figsize=(4, 4))

    steps_x = np.arange(0, steps)
    ax.plot(steps_x, loss_fmn_base, label='FMN Base loss')
    ax.plot(steps_x, fmn_vec.attack_data['loss'], label='FMN Vec loss', linestyle='dotted')

    ax.grid()
    ax.legend()

    fig_path = os.path.join(exp_path, f"SchedulerVecExps/FMN_vecVsBase_lossData"
                                      f"_{optimizer}_steps{steps}_batch{batch_size}_eps_{epsilon_name}.pdf")
    fig.savefig(fig_path)
'''

if __name__ == '__main__':
    plot_loss_AA_fmn_comparison()
