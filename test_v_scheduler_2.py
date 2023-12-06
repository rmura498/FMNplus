import pickle
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

if __name__ == '__main__':
    batch_size = 50
    num_batches = 1
    steps = 20
    loss = 'CE'
    optimizer = 'Adam'
    norm = inf # same as float('inf')
    alpha_init = 1
    alpha_final = None

    # loading model and dataset
    model, dataset, model_name, dataset_name = load_data(model_id=8)
    model.eval().to(device)

    # Dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Instantiating FMN vec
    fmn_vec = FMNVec(
        model,
        steps=steps,
        loss=loss,
        device=device,
        epsilon=8/255,
        optimizer=optimizer,
        norm=norm,
        alpha_init=alpha_init,
        alpha_final=alpha_final
    )

    fmn_base = FMNBase(
        model,
        steps=steps,
        loss=loss,
        device=device,
        epsilon=8 / 255,
        optimizer=optimizer,
        scheduler='RLROP',
        norm=norm,
        alpha_init=alpha_init,
        alpha_final=alpha_final
    )

    loss_fmn_base = []

    for i, (samples, labels) in enumerate(dataloader):
        # clean misclassified
        logits = model(samples)
        pred_labels = logits.argmax(dim=1)
        correctly_classified_samples = pred_labels == labels
        samples = samples[correctly_classified_samples]
        labels = labels[correctly_classified_samples]

        _, _ = fmn_vec.forward(samples, labels)

        print("\n\nRunning FMN base\n")

        # running multiple FMNBase instances
        # re-instantiating FMN base
        '''
        for j, (sample, label) in enumerate(zip(samples, labels)):
            sample.unsqueeze_(0)
            label.unsqueeze_(0)

            print(f"Sample, label shapes at [{i}]:\n{sample.shape}\n{label}")
            fmn_base = FMNBase(
                model,
                steps=steps,
                loss=loss,
                device=device,
                epsilon=8 / 255,
                optimizer=optimizer,
                scheduler='RLROP',
                norm=norm,
                alpha_init=alpha_init,
                alpha_final=alpha_final
            )
            _, _ = fmn_base.forward(sample, label)
            loss_fmn_base.append(fmn_base.attack_data['loss'])
            print(f"Cur loss len: {len(fmn_base.attack_data['loss'])}")
        '''

        _, _ = fmn_base.forward(samples, labels)

        if i+1 == num_batches: break

    # loss_fmn_base = torch.tensor(loss_fmn_base).unsqueeze(0).mean(dim=1).squeeze(0).tolist()
    loss_fmn_base = fmn_base.attack_data['loss']
    loss_data = {
        'loss_fmn_base': loss_fmn_base,
        'loss_fmn_vec': fmn_vec.attack_data['loss']
    }

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y")
    filename = (f"SchedulerVecExps/FMN_vecVsBase_lossData_{formatted_date}"
                f"_{optimizer}_steps{steps}_batch{batch_size}.pkl")

    with open(filename, 'wb') as file:
        pickle.dump(loss_data, file)

    # plot losses
    fig, ax = plt.subplots(figsize=(4, 4))

    steps_x = np.arange(0, steps)
    ax.plot(steps_x, loss_fmn_base, label='FMN Base loss')
    ax.plot(steps_x, fmn_vec.attack_data['loss'], label='FMN Vec loss', linestyle='dotted')

    ax.grid()
    ax.legend()

    # Get current date
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y")
    fig.savefig(f"SchedulerVecExps/FMN_vecVsBase_lossData_{formatted_date}"
                f"_{optimizer}_steps{steps}_batch{batch_size}.pdf")



