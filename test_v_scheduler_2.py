'''
07/12/2023
Raffaele Mura, Giuseppe Floris, Luca Scionis

Testing multiple FMN configuration against AutoAttack

'''

import os, pickle, argparse
from datetime import datetime

import torch
from torch import inf
from torch.utils.data import DataLoader

from Attacks.fmn_base_vec import FMN as FMNVec
from Attacks.fmn_base import FMN as FMNBase
from Utils.load_model import load_data

from autoattack.autoattack import AutoAttack


parser = argparse.ArgumentParser(description="Perform multiple attacks using FMN and AA.")

parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--num_batches', type=int, default=1, help='Number of batches')
parser.add_argument('--steps', type=int, default=30, help='Number of steps')
parser.add_argument('--epsilon', type=float, default=8/255, help='Epsilon value, None for dynamic one')
parser.add_argument('--loss', type=str, default='CE', choices=['CE', 'DLR'], help='Loss function')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], help='Optimizer')
parser.add_argument('--scheduler', type=str, default='RLROP', choices=['RLROP', 'CALR'], help='Scheduler')
parser.add_argument('--norm', type=float, default=float('inf'), help='Norm value')
parser.add_argument('--model_id', type=int, default=8, help='Model ID')
parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle data')
parser.add_argument('--attack_type', type=str, default='FMNBase', choices=['FMNBase', 'FMNVec', 'AA'], help='Type of attack')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device to use (cuda:0, cuda:1) - int')


def configure_autoattack(model, steps, loss='CE'):
    _attacks = {
        'CE': ('apgd-ce',),
        'DLR': ('apgd-dlr',)
    }

    adversary = AutoAttack(
        model,
        norm='Linf',
        eps=8/255,
        version='standard',
        device=device,
        verbose=True
    )
    adversary.attacks_to_run = _attacks[loss]
    adversary.apgd.n_restarts = 1
    adversary.apgd.n_iter = steps

    return adversary


def main(
        batch_size = 10,
        num_batches = 1,
        steps = 30,
        epsilon = 8/255, # None for dynamic one
        loss='CE',
        optimizer = 'Adam',
        scheduler = 'RLROP',
        norm = inf,  # same as float('inf')
        model_id = 8,
        shuffle=True,
        attack_type='FMNBase'
):
    # creating exp folder
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y%H")
    epsilon_name = "8-255" if epsilon == 8 / 255 else "None"
    exp_path = os.path.join("Exps", formatted_date, f"{attack_type}-eps{epsilon_name}"
                                                    f"-bs{batch_size}-steps{steps}-loss{loss}")

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # loading model and dataset
    model, dataset, model_name, dataset_name = load_data(model_id=model_id)
    model.eval().to(device)

    _bs = batch_size * 2
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=_bs,
        shuffle=shuffle
    )

    if attack_type == 'AA':
        attack = configure_autoattack(model, steps, loss)
    elif attack_type == 'FMNVec':
        attack = FMNVec(
            model=model,
            steps=steps,
            loss=loss,
            device=device,
            epsilon=epsilon,
            optimizer=optimizer,
            norm=norm
        )
    else:
        attack = FMNBase(
            model=model,
            steps=steps,
            loss=loss,
            device=device,
            epsilon=epsilon,
            optimizer=optimizer,
            scheduler=scheduler,
            norm=norm
        )

    for i, (samples, labels) in enumerate(dataloader):
        print(f"Cleaning misclassified on batch {i}")
        # clean misclassified
        logits = model(samples)
        pred_labels = logits.argmax(dim=1)
        correctly_classified_samples = pred_labels == labels
        samples = samples[correctly_classified_samples]
        labels = labels[correctly_classified_samples]

        # retrieving only requested batch size
        samples = samples[:batch_size]
        labels = labels[:batch_size]

        print(f"Running attack on batch {i}")
        if 'AA' in attack_type:
            attack.run_standard_evaluation(samples, labels, bs=batch_size)
            loss_data = attack.apgd.loss_total
            sr_data = attack.success_rate
        else:
            attack.forward(images=samples, labels=labels)
            loss_data = attack.attack_data['loss']
            sr_data = attack.attack_data['success_rate']

        attack_data = {
            'loss': loss_data,
            'success_rate': sr_data
        }

        # Saving data
        print(f"Saving attack data on batch {i}")
        filename = (f"{formatted_date}"
                    f"_{optimizer}_steps{steps}_batch{batch_size}.pkl")

        with open(os.path.join(exp_path, filename), 'wb') as file:
            pickle.dump(attack_data, file)

        if i+1 == num_batches: break


if __name__ == '__main__':
    # retrieve parsed arguments
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    num_batches = int(args.num_batches)
    steps = int(args.steps)
    epsilon = float(args.epsilon)
    loss = str(args.loss)
    optimizer = str(args.optimizer)
    scheduler = str(args.scheduler)
    norm = float(args.norm)
    model_id = int(args.model_id)
    shuffle = bool(args.shuffle)
    attack_type = str(args.attack_type)
    cuda_device = int(args.cuda_device)

    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed('42')

    main(
        batch_size=batch_size,
        num_batches=num_batches,
        steps=steps,
        epsilon=epsilon,  # None for dynamic one
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        norm=norm,  # same as float('inf')
        model_id=model_id,
        shuffle=shuffle,
        attack_type=attack_type
    )
