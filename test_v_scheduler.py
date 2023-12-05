import pickle
from datetime import datetime

import torch
from torch import inf
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from Attacks.fmn_base_vec import FMN as FMNVec
from Utils.load_model import load_data

from autoattack.autoattack import AutoAttack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    batch_size = 50
    num_batches = 1
    steps = 40
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
    fmn = FMNVec(
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

    for i, (samples, labels) in enumerate(dataloader):
        # clean misclassified
        logits = model(samples)
        pred_labels = logits.argmax(dim=1)
        correctly_classified_samples = pred_labels == labels
        samples = samples[correctly_classified_samples]
        labels = labels[correctly_classified_samples]

        best_adv, _ = fmn.forward(samples, labels)

        # running autoattack/
        adversary = AutoAttack(
            model,
            norm='Linf',
            eps=8/255,
            version='standard',
            device=device,
            verbose=False
        )
        adversary.attacks_to_run = ['apgd-ce',]
        adversary.apgd.n_restarts = 1
        adversary.apgd.n_iter = steps

        adversary.run_standard_evaluation(samples, labels, bs=batch_size)

        if i+1 == num_batches: break

    loss_data = {
        'loss_aa': -adversary.apgd.loss_total,
        'loss_fmn': fmn.attack_data['loss']
    }

    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y")
    filename = (f"SchedulerVecExps/FMN_AA_lossData_{formatted_date}"
                f"_{optimizer}_steps{steps}_batch{batch_size}.pkl")

    with open(filename, 'wb') as file:
        pickle.dump(loss_data, file)

    # TODO: add success rate


