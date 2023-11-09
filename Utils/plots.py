import torch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from Utils.metrics import compute_robust


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
                        best_distances, aa_robust):
    if len(distance) == 0:
        print("Error: Not enough distances per experiment!")
        return

    fig, ax = plt.figure(figsize=(3,3))


    # single experiment
    steps = len(distance)

    distances, robust_per_iter = compute_robust(distance, best_distances)

    distances = np.array(distances)
    distances.sort()
    robust_per_iter.sort(reverse=True)

    ax.plot(distances,
            robust_per_iter)
    ax.plot(8/255, aa_robust, 'x', label='AA')
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