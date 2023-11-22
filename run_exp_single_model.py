import os, pickle, math, argparse

import torch

from Attacks.fmn import FMN
from Utils.load_model import load_dataset, load_model, load_data
from Utils.plots import plot_distance
from Utils.fmn_strategies import fmn_strategies
from Utils.comparing_strategies import evaluate_strategies

from config import MODEL_DATASET, EXP_DIRECTORY

from robustbench.utils import clean_accuracy

"""
FMN parametric strategy 
- the loss we're optimizing for the attack (CE, LL, DLR)
- Initial point that we use for starting the attack 
- The optimizer that we use 
- the scheduler 
- Decide how we transform the gradient in each step. (normalization or linear projection into the step size)  

"""

parser = argparse.ArgumentParser(description='Perform multiple attacks using FMN+, a parametric version of FMN.')
parser.add_argument('-mn', '--model_numbers',
                    default=0,
                    help='How many models you want to test.')
parser.add_argument('-s', '--steps',
                    default=30,
                    help='Provide the number of steps of a single attack.')
parser.add_argument('-bs', '--batch_size',
                    default=10,
                    help='Provide the batch size.')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiments(model_id=0, steps=30, batch_size=10):
    # model definition
    model, dataset, model_name, dataset_name = load_data(model_id=model_id)
    model.eval()
    model.to(device)

    # retrieving FMN set of configurations
    fmn_dict = fmn_strategies()

    # dataset definition
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    samples = torch.empty([1, 3, 32, 32])
    labels = torch.empty([1], dtype=torch.int64)
    i = 0

    for sample, label in dataloader:
        y_pred = model(sample.reshape(1, 3, 32, 32))
        y_pred = torch.argmax(y_pred)

        if y_pred == label:
            samples = torch.cat([samples, sample])
            labels = torch.cat([labels, label])
            i += 1
        if i >= batch_size:
            break
    # print(samples)
    samples = samples[1:]
    labels = labels[1:]

    # print(labels.shape)
    clean_acc = clean_accuracy(model, samples, labels)
    print("clean accuracy", clean_acc)
    # print(labels)

    for i in range(len(fmn_dict)):
        print(i)
        print("Running:", fmn_dict[str(i)])
        attack = FMN(model, steps=steps, loss=fmn_dict[str(i)]['loss'], optimizer=fmn_dict[str(i)]['optimizer'],
                     scheduler=fmn_dict[str(i)]['scheduler'], gradient_strategy=fmn_dict[str(i)]['gradient_strategy'],
                     initialization_strategy=fmn_dict[str(i)]['initialization_strategy'], device=device)
        adv_x, best_distance = attack.forward(samples, labels)

        robust_acc = clean_accuracy(model, adv_x, labels, device=device)
        print(f"Robust Accuracy:{fmn_dict[str(i)]}", robust_acc)
        attack_data = attack.attack_data

        current_exp_dir = os.path.join(EXP_DIRECTORY, model_name, dataset_name)
        if not os.path.exists(current_exp_dir): os.makedirs(current_exp_dir)
        filename = os.path.join(current_exp_dir,
                                f'{fmn_dict[str(i)]["optimizer"]}-{fmn_dict[str(i)]["scheduler"]}-'
                                f'{fmn_dict[str(i)]["loss"]}-{fmn_dict[str(i)]["gradient_strategy"]}-'
                                f'{fmn_dict[str(i)]["initialization_strategy"]}.pkl')
        # Save the object to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(attack_data, file)

    # evaluate_strategies(batch_size)


if __name__ == '__main__':
    model_numbers = int(args.model_numbers)
    steps = int(args.steps)
    batch_size = int(args.batch_size)

    if model_numbers > len(MODEL_DATASET.keys()):
        model_numbers = len(MODEL_DATASET.keys())

    for i in range(6, 8):
        run_experiments(model_id=i, steps=steps, batch_size=batch_size)
