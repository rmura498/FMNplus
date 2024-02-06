'''
07/12/2023
Raffaele Mura, Giuseppe Floris, Luca Scionis

Testing multiple FMN configuration against AutoAttack

'''

import os, pickle, argparse

import torch
from torch.utils.data import DataLoader

from Utils.load_model import load_data

from config import CLEANED_SAMPLES_DIRECTORY


parser = argparse.ArgumentParser(description="Perform multiple attacks using FMN and AA.")

parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--num_batches', type=int, default=1, help='Number of batches')
parser.add_argument('--model_id', type=int, default=8, help='Model ID')
parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle data')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device to use (cuda:0, cuda:1) - int')


def main(
        batch_size=10,
        num_batches=1,
        model_id=8,
        shuffle=False
):
    # creating exp folder
    exp_path = os.path.join(CLEANED_SAMPLES_DIRECTORY, f"mid{model_id}_bs{batch_size}_nb{num_batches}")

    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)

    # loading model and dataset
    model, dataset, model_name, dataset_name = load_data(model_id=model_id)
    model.eval()
    model = model.to(device)

    _bs = batch_size + 250
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=_bs,
        shuffle=shuffle
    )

    for i, (samples, labels) in enumerate(dataloader):
        print(f"Cleaning misclassified on batch {i}")
        # clean misclassified
        samples = samples.to(device)
        labels = labels.to(device)
        logits = model(samples)
        pred_labels = logits.argmax(dim=1)
        correctly_classified_samples = pred_labels == labels
        samples = samples[correctly_classified_samples]
        labels = labels[correctly_classified_samples]

        # retrieving only requested batch size
        samples = samples[:batch_size]
        labels = labels[:batch_size]

        print(f"\tShape of samples:\n{samples.shape}")
        print(f"\tShape of labels:\n{labels.shape}")

        # Saving data
        print(f"Saving samples, labels of batch {i}")

        torch.save(samples, os.path.join(exp_path, f"samples_b{i}.pt"))
        torch.save(labels, os.path.join(exp_path, f"labels_b{i}.pt"))

        if i+1 == num_batches: break


if __name__ == '__main__':
    # retrieve parsed arguments
    args = parser.parse_args()

    batch_size = int(args.batch_size)
    num_batches = int(args.num_batches)
    model_id = int(args.model_id)
    shuffle = bool(args.shuffle)
    cuda_device = int(args.cuda_device)

    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    main(
        batch_size=batch_size,
        num_batches=num_batches,
        model_id=model_id,
        shuffle=shuffle
    )
