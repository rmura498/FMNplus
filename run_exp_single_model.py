import os, pickle, math, argparse

import torch

from Attacks.fmn_single_distance_estimation import FMN as FMN_single_dist_est
from Attacks.fmn_base import FMN as FMN_base
from Attacks.fmn_base_test import FMN as FMN_base_test
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
parser.add_argument('-f_mid', '--first_mid',
                    default=0,
                    help='The first model id to test.')
parser.add_argument('-m_num', '--models_number',
                    default=1,
                    help='How many models to test.')
parser.add_argument('-s', '--steps',
                    default=30,
                    help='Provide the number of steps of a single attack.')
parser.add_argument('-bs', '--batch_size',
                    default=10,
                    help='Provide the batch size.')

parser.add_argument('-en', '--exp_name', default='base')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_experiments(model_id=0, steps=30, batch_size=10, exp_name='base'):
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

    attack_type = FMN_base_test
    def first_strategies():
        for i in range(len(fmn_dict)//2):
            print(i)
            print("Running:", fmn_dict[str(i)])
            attack = attack_type(model, steps=steps, loss=fmn_dict[str(i)]['loss'], optimizer=fmn_dict[str(i)]['optimizer'],
                         scheduler=fmn_dict[str(i)]['scheduler'], gradient_strategy=fmn_dict[str(i)]['gradient_strategy'],
                         initialization_strategy=fmn_dict[str(i)]['initialization_strategy'], device=device)
            adv_x, best_distance = attack.forward(samples, labels)

            robust_acc = clean_accuracy(model, adv_x, labels, device=device)
            print(f"Robust Accuracy:{fmn_dict[str(i)]}", robust_acc)
            attack_data = attack.attack_data

            current_exp_dir = os.path.join(EXP_DIRECTORY, f"{model_name}_{str(attack_type.__class__.__name__)}", dataset_name)
            if not os.path.exists(current_exp_dir): os.makedirs(current_exp_dir)
            filename = os.path.join(current_exp_dir,
                                    f'{fmn_dict[str(i)]["optimizer"]}-{fmn_dict[str(i)]["scheduler"]}-'
                                    f'{fmn_dict[str(i)]["loss"]}-{fmn_dict[str(i)]["gradient_strategy"]}-'
                                    f'{fmn_dict[str(i)]["initialization_strategy"]}.pkl')
            # Save the object to a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(attack_data, file)

    def second_strategies():
        for i in range(len(fmn_dict) // 2, len(fmn_dict)):
            print(i)
            print("Running:", fmn_dict[str(i)])
            attack = attack_type(model, steps=steps, loss=fmn_dict[str(i)]['loss'], optimizer=fmn_dict[str(i)]['optimizer'],
                         scheduler=fmn_dict[str(i)]['scheduler'],
                         gradient_strategy=fmn_dict[str(i)]['gradient_strategy'],
                         initialization_strategy=fmn_dict[str(i)]['initialization_strategy'], device=device)
            adv_x, best_distance = attack.forward(samples, labels)

            robust_acc = clean_accuracy(model, adv_x, labels, device=device)
            print(f"Robust Accuracy:{fmn_dict[str(i)]}", robust_acc)
            attack_data = attack.attack_data

            current_exp_dir = os.path.join(EXP_DIRECTORY, f"{model_name}_{exp_name}", dataset_name, )
            if not os.path.exists(current_exp_dir): os.makedirs(current_exp_dir)
            filename = os.path.join(current_exp_dir,
                                    f'{fmn_dict[str(i)]["optimizer"]}-{fmn_dict[str(i)]["scheduler"]}-'
                                    f'{fmn_dict[str(i)]["loss"]}-{fmn_dict[str(i)]["gradient_strategy"]}-'
                                    f'{fmn_dict[str(i)]["initialization_strategy"]}.pkl')
            # Save the object to a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(attack_data, file)

    def run_single_strategy(strategy):
        print("Running:", strategy)
        attack = attack_type(model, steps=steps, loss=strategy['loss'], optimizer=strategy['optimizer'],
                             scheduler=strategy['scheduler'],
                             gradient_strategy=strategy['gradient_strategy'],
                             initialization_strategy=strategy['initialization_strategy'], device=device)
        adv_x, best_distance = attack.forward(samples, labels)

        robust_acc = clean_accuracy(model, adv_x, labels, device=device)
        print(f"Robust Accuracy:{strategy}", robust_acc)
        attack_data = attack.attack_data

        current_exp_dir = os.path.join(EXP_DIRECTORY, f"{model_name}_{exp_name}",
                                       dataset_name, )
        if not os.path.exists(current_exp_dir): os.makedirs(current_exp_dir)
        filename = os.path.join(current_exp_dir,
                                f'{strategy["optimizer"]}-{strategy["scheduler"]}-'
                                f'{strategy["loss"]}-{strategy["gradient_strategy"]}-'
                                f'{strategy["initialization_strategy"]}.pkl')
        # Save the object to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(attack_data, file)

    import threading

    workers = []
    for i in range(len(fmn_dict)):
        t = threading.Thread(target=run_single_strategy, args=(fmn_dict[str(i)],))
        t.start()
        workers.append(t)
        break

    for w in workers:
        w.join()
    # evaluate_strategies(batch_size)


if __name__ == '__main__':
    first_mid = int(args.first_mid)
    models_number = int(args.models_number)
    steps = int(args.steps)
    batch_size = int(args.batch_size)
    exp_name = args.exp_name

    if first_mid+models_number > len(MODEL_DATASET.keys()):
        models_number = len(MODEL_DATASET.keys()) - (first_mid+models_number)

    for i in range(first_mid, first_mid+models_number):
        run_experiments(model_id=i, steps=steps, batch_size=batch_size, exp_name=exp_name)
