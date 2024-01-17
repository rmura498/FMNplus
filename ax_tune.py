import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from ax.service.ax_client import AxClient

from Attacks.fmn_base import FMN
from Utils.load_model import load_data
from Tuning.AxTuning.search_space import OptimizerParams, SchedulerParams

parser = argparse.ArgumentParser(description="Tune FMN Hyperparameters with Ax")

parser.add_argument('--model_id', type=int, default=8, help='Robustbench model\'s id')
parser.add_argument('--batch_size', type=int, default=32, help='Size of a single batch')
parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle samples of each batch')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'Adamax'], help='Optimizer for the attack')
parser.add_argument('--scheduler', type=str, default=None, choices=['RLROPVec', 'CALR', 'None'],  help='Scheduler for the attack')
parser.add_argument('--steps', type=int, default=20, help='Steps of the attack')
parser.add_argument('--loss', type=str, default='CE', help='Loss for the attack')
parser.add_argument('--norm', type=float, default=float('inf'), help='Type of norm (e.g. Linf, L0, L1, L2)')
parser.add_argument('--gradient_update', type=str, default='Sign', choices=['Normalization', 'Projection', 'Sign'], help='Attack\'s gradient update strategy')
parser.add_argument('--n_trials', type=int, default=1, help='How many hyperparams optimization trials')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use (cpu, cuda:0, cuda:1)')
parser.add_argument('--cuda_device', type=int, default=0, help='Specific gpu to use like 0, 1 etc - (optional)')


args = parser.parse_args()

model_id = int(args.model_id)
batch_size = int(args.batch_size)
shuffle = bool(args.shuffle)
optimizer = args.optimizer
scheduler = str(args.scheduler)
steps = int(args.steps)
loss = args.loss
norm = float(args.norm)
gradient_update = args.gradient_update
n_trials = int(args.n_trials)

if scheduler == 'None': scheduler = None

device = args.device
if not torch.cuda.is_available():
    device = 'cpu'


def attack_evaluate(parametrization):
    optimizer_config = {k: parametrization[k] for k in set(opt_params)}
    scheduler_config = None
    if scheduler is not None:
        scheduler_config = {k: parametrization[k] for k in set(sch_params)}

    attack = FMN(
        model=model,
        steps=steps,
        loss=loss,
        device=device,
        epsilon=None,
        optimizer=optimizer,
        scheduler=scheduler,
        norm=norm,
        gradient_strategy=gradient_update,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config
    )

    _, best_distance, _ = attack.forward(images=images, labels=labels)
    best_distance = float(best_distance.mean().item())
    # sr = float(sr)

    return {'distance': (best_distance, 0.0)}


# Retrieve optimizer and scheduler params
opt_params = OptimizerParams[optimizer]
sch_params = None
if scheduler is not None:
    sch_params = SchedulerParams[scheduler]
    if 'T_max' in sch_params:
        sch_params['T_max'] = sch_params['T_max'](steps)
    if 'batch_size' in sch_params:
        sch_params['batch_size'] = sch_params['batch_size'](batch_size)

# Loading model and dataset
print("\t[Tuning] Retrieving the model and the dataset...")
model, dataset, model_name, dataset_name = load_data(model_id=model_id)
model.eval()
model = model.to(device)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle
)
images, labels = next(iter(dataloader))

# Create Ax Experiment
print("\t[Tuning] Creating the Ax client and experiment...")
ax_client = AxClient()

# Defining the Search Space
if scheduler is not None:
    params = list(opt_params.values()) + list(sch_params.values())
else:
    params = list(opt_params.values())

# Create an experiment with required arguments: name, parameters, and objective_name.
experiment_name = f'mid{model_id}_{batch_size}_{steps}_{optimizer}"\
                    f"_{scheduler}_{loss}_{gradient_update}'
ax_client.create_experiment(
    name=experiment_name,
    parameters=params,
    objective_name="distance",
    minimize=True
)

print("\t[Tuning] Starting the Hyperparameters Optimization...")
for i in range(n_trials):
    print("\t[Tuning] Running trial {}")
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=attack_evaluate(parameters))

print("\t[Tuning] Finished the Hyperparameters Optimization; printing the trials list and best parameters...")
print(ax_client.get_trials_data_frame())

best_parameters, values = ax_client.get_best_parameters()
print("\t[Tuning] Best parameters: ", best_parameters)

print("\t[Tuning] Saving the experiment data...")
current_date = datetime.now()
formatted_date = current_date.strftime("%d%m%y%H")
ax_client.save_to_json_file(filepath=f'{formatted_date}_{experiment_name}.json')