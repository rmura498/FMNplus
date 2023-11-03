import torch
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def NormalizeData(data):
    return 2*((data - np.min(data)) / (np.max(data) - np.min(data))) -1
    #return ((data - np.min(data)) / (np.max(data) - np.min(data)))


def plot_distance():
    filenames = os.listdir('./experiments/')

    fig, axs = plt.subplots(4, 3, figsize=(8,10), layout='constrained')

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
        epsilons = NormalizeData(epsilon_sample)
        distances = NormalizeData(distance_sample)
        loss = NormalizeData(loss_sample)
        steps = len(epsilons)

        ax.plot(epsilons, label='epsilon')
        ax.plot(distances, label='distances')
        ax.plot(loss, label='loss')
        ax.legend(loc=0, prop={'size': 4})

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4

        title = filename.split('/')

        ax.set_title(f"Steps: {steps}, S: {title[-1]}, MiDist: {min_dist}", fontsize=fontsize)

        ax.set_xlabel("Steps")
        ax.set_ylabel("Distance - Loss")
        ax.grid()

    plt.show()
    fig.savefig("example.pdf")
