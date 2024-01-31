from datetime import datetime

import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import inf

if __name__ == '__main__':
    file_path = '../SchedulerVecExps/FMN_AA_lossData_051223_Adam_steps40_batch50.pkl'

    with open(file_path, 'rb') as file:
        loss_data = pickle.load(file)

    loss_aa = loss_data['loss_aa']
    loss_fmn = loss_data['loss_fmn']
    print(f"AutoAttack Loss:\n{loss_aa}\nlen: {len(loss_aa)}")
    print(f"FMN Loss:\n{loss_fmn}\nlen: {len(loss_fmn)}")

    steps = len(loss_aa)

    # plot losses
    fig, ax = plt.subplots(figsize=(4, 4))

    steps_x = np.arange(0, steps)
    ax.plot(steps_x, loss_aa, label='AA mean loss')
    ax.plot(steps_x, loss_fmn, label='FMN mean loss')

    ax.grid()
    ax.legend()

    # Get current date
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y")
    fig.savefig(f"SchedulerVecExps/FMN_vec_loss_comparison_{formatted_date}_vecScheduler_steps{steps}.pdf")