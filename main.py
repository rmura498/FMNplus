import torch
import math
import os
import pickle
from Attacks.fmn import FMN
import numpy as np
from Utils.load_model import load_dataset, load_model
from robustbench.utils import clean_accuracy
from Utils.plots import plot_distance
from Utils.fmn_strategies import fmn_strategies
from Utils.comparing_strategies import evaluate_strategies

"""
FMN parametric strategy 
- the loss we're optimizing for the attack (CE, LL, DLR)
- Initial point that we use for starting the attack 
- The optimizer that we use 
- the scheduler 
- Decide how we transform the gradient in each step. (normalization or linear projection into the step size)  

"""


def run_experiments(steps=30, batch_size=10, batch_number=1):
    # model definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model('Gowal2021Improving_R18_ddpm_100m', 'cifar10')
    dataset = load_dataset('cifar10')
    model.eval()
    model.to(device)

    # retrieving FMN set of configurations
    fmn_dict = fmn_strategies()

    # dataset definition
    dataset_frac = list(range(math.floor(len(dataset) * 0.5) + 1, len(dataset)))
    dataset_frac = torch.utils.data.Subset(dataset, dataset_frac)
    dl_test = torch.utils.data.DataLoader(dataset_frac, batch_size=1, shuffle=False)

    samples = torch.empty([1, 3, 32, 32])
    labels = torch.empty([1], dtype=torch.int64)
    i = 0


    for sample, label in dl_test:
        y_pred = model(sample.reshape(1, 3, 32, 32))
        y_pred = torch.argmax(y_pred)

        if y_pred == label:
            samples = torch.cat([samples, sample])
            labels = torch.cat([labels, label])
            i += 1
        if i >= batch_size:
            break
    print(samples)
    samples = samples[1:]
    labels = labels[1:]

    print(labels.shape)
    clean_acc = clean_accuracy(model, samples, labels)
    print("clean accuracy", clean_acc)
    print(labels)

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

        directory = './experiments/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory,
                                f'{fmn_dict[str(i)]["optimizer"]}-{fmn_dict[str(i)]["scheduler"]}-'
                                f'{fmn_dict[str(i)]["loss"]}-{fmn_dict[str(i)]["gradient_strategy"]}-'
                                f'{fmn_dict[str(i)]["initialization_strategy"]}.pkl')
        # Save the object to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(attack_data, file)

    evaluate_strategies(batch_size)


run_experiments()
# plot_distance()
