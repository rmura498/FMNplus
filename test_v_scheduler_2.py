import os, pickle
from datetime import datetime

import torch
from torch import inf
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from Attacks.fmn_base_vec import FMN as FMNVec
from Attacks.fmn_base import FMN as FMNBase
from Utils.load_model import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
        batch_size = 10,
        num_batches = 1,
        steps = 30,
        epsilon = 8/244, # None for dynamic one
        loss='CE',
        optimizer = 'Adam',
        scheduler = 'RLROP',
        norm = inf,  # same as float('inf')
        alpha_init = 1,
        alpha_final = None,
        verbose = True,
        model_id = 8,
        plot = False,
        shuffle=True
):

    # loading model and dataset
    model, dataset, model_name, dataset_name = load_data(model_id=model_id)
    model.eval().to(device)

    # Dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    # Instantiating FMN vec
    fmn_vec = FMNVec(
        model,
        steps=steps,
        loss=loss,
        device=device,
        epsilon=epsilon,
        optimizer=optimizer,
        norm=norm,
        alpha_init=alpha_init,
        alpha_final=alpha_final,
        verbose=verbose
    )

    fmn_base = FMNBase(
        model,
        steps=steps,
        loss=loss,
        device=device,
        epsilon=epsilon,
        optimizer=optimizer,
        scheduler=scheduler,
        norm=norm,
        alpha_init=alpha_init,
        alpha_final=alpha_final,
        verbose=verbose
    )

    for i, (samples, labels) in enumerate(dataloader):
        # clean misclassified
        logits = model(samples)
        pred_labels = logits.argmax(dim=1)
        correctly_classified_samples = pred_labels == labels
        samples = samples[correctly_classified_samples]
        labels = labels[correctly_classified_samples]

        print("\n\nRunning FMN vec\n")
        _, _ = fmn_vec.forward(samples, labels)
        print("\n\nRunning FMN base\n")
        _, _ = fmn_base.forward(samples, labels)

        if i+1 == num_batches: break

    # loss_fmn_base = torch.tensor(loss_fmn_base).unsqueeze(0).mean(dim=1).squeeze(0).tolist()
    loss_fmn_base = fmn_base.attack_data['loss']
    loss_data = {
        'loss_fmn_base': loss_fmn_base,
        'loss_fmn_vec': fmn_vec.attack_data['loss']
    }

    print(f"\nFMN Base loss:\n{loss_fmn_base}")
    print(f"FMN Vec loss:\n{fmn_vec.attack_data['loss']}")

    epsilon_name = "8-255" if epsilon == 8/255 else "None"
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y")

    exp_path = os.path.join("SchedulerVecExps", formatted_date)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    filename = os.path.join(exp_path, f"FMN_vecVsBase_lossData_{optimizer}_steps{steps}"
                                      f"_batch{batch_size}_eps_{epsilon_name}.pkl")

    with open(filename, 'wb') as file:
        pickle.dump(loss_data, file)

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


if __name__ == '__main__':
    main(batch_size=5, steps=50)
