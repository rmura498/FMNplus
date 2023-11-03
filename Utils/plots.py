import torch
import matplotlib.pyplot as plt
import numpy as np


def NormalizeData(data):
    return 2*((data - np.min(data)) / (np.max(data) - np.min(data))) -1


def plot_distance(distances, epsilons, loss, loss_type):
    min_dist = distances[-1]
    #print(distances)
    epsilons = NormalizeData(epsilons)
    distances = NormalizeData(distances)
    loss = NormalizeData(loss)
    steps = len(epsilons)
    fig, ax = plt.subplots()
    ax.plot(epsilons, label='epsilon')
    ax.plot(distances, label='distances')
    ax.plot(loss, label='loss')
    ax.legend(loc=0, prop={'size': 8})

    dpi = fig.dpi
    rect_height_inch = ax.bbox.height / dpi
    fontsize = rect_height_inch * 4

    ax.set_title(f"Steps: {steps}, loss: {loss_type}, Minimum Distance: {min_dist}", fontsize=fontsize)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Distance - Loss")
    ax.grid()

    plt.show()


def plot_epsilon_robust(exps_distances=[],
                        exps_names=[],
                        exps_params=[],
                        best_distances=[]):
    if len(exps_distances) == 0:
        print("Error: Not enough distances per experiment!")
        return

    # number of experiments
    n_exps = len(exps_distances)
    plot_grid_size = n_exps // 2 + 1

    fig = plt.figure(figsize=(3, 3))

    for i, exp_distances in enumerate(exps_distances):
        # single experiment
        steps = len(exp_distances)
        batch_size = len(exp_distances[0])

        distances, robust_per_iter = compute_robust(exp_distances, best_distances[i])

        distances = np.array(distances)
        distances.sort()
        robust_per_iter.sort(reverse=True)

        ax = fig.add_subplot(plot_grid_size, plot_grid_size, i + 1)
        ax.plot(distances,
                robust_per_iter)
        ax.plot(8 / 255, 0.661, 'x', label='AA')
        ax.grid()

        dpi = fig.dpi
        rect_height_inch = ax.bbox.height / dpi
        fontsize = rect_height_inch * 4
        ax.set_title(f"Steps: {steps}, batch: {batch_size}, norm: {exps_params[i]['norm']},"
                     f"\nOptimizer: {exps_params[i]['optimizer']}, Scheduler: {exps_params[i]['scheduler']}",
                     fontsize=fontsize)
        ax.legend(loc=0, prop={'size': 8})
        ax.set_xlabel(r"$||\boldsymbol{\delta}||_\infty$")
        ax.set_ylabel("R. Acc.")

    plt.xlim([0, 0.2])
    fig.savefig("example.pdf")
