import os, argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from Attacks.fmn_base import FMN
from Utils.load_model import load_data

from Tuning.AxTuning.search_space import OptimizerParams, SchedulerParams


parser = argparse.ArgumentParser(description="Run FMN Baseline attacks")

parser.add_argument('--model_id', type=int, default=8, help='Robustbench model\'s id')
parser.add_argument('--batch_size', type=int, default=128, help='Size of a single batch')
parser.add_argument('--n_batches', type=int, default=32, help='N batches')
parser.add_argument('--shuffle', action="store_true", help='Shuffle samples of each batch')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'Adamax'], help='Optimizer for the attack')
parser.add_argument('--scheduler', type=str, default=None, choices=['RLROPVec', 'CALR', 'None'],  help='Scheduler for the attack')
parser.add_argument('--steps', type=int, default=20, help='Steps of the attack')
parser.add_argument('--loss', type=str, default='CE', help='Loss for the attack')
parser.add_argument('--norm', type=float, default=float('inf'), help='Type of norm (e.g. Linf, L0, L1, L2)')
parser.add_argument('--gradient_update', type=str, default='Sign', choices=['Normalization', 'Projection', 'Sign'], help='Attack\'s gradient update strategy')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use (cpu, cuda:0, cuda:1)')
parser.add_argument('--cuda_device', type=int, default=-1, help='Specific gpu to use like -1 (discard gpu selection), 0 or 1')
parser.add_argument('--parallelize', action="store_true", help='Parallelize the model among the GPUs')
parser.add_argument('--best_params_path', type=str, default='none', help="Best params path (e.g. './best_params.pth'")

args = parser.parse_args()

model_id = int(args.model_id)
batch_size = int(args.batch_size)
n_batches = int(args.n_batches)
shuffle = args.shuffle
optimizer = args.optimizer
scheduler = str(args.scheduler)
steps = int(args.steps)
loss = args.loss
norm = float(args.norm)
gradient_update = args.gradient_update
device = args.device
cuda_device = int(args.cuda_device)
parallelize = args.parallelize
best_params_path = args.best_params_path if args.best_params_path != 'none' else None

if scheduler == 'None': scheduler = None

if not torch.cuda.is_available():
    device = 'cpu'


if device == 'cuda' and cuda_device != -1 and not parallelize:
    device = 'cuda:' + str(cuda_device)
elif parallelize and device == 'cuda':
    device = torch.device(device)

current_date = datetime.now()
formatted_date = current_date.strftime("%d%m%y%H")
experiment_name = f'FMN_attack_mid{model_id}_{batch_size}_{steps}_{optimizer}_{scheduler}_{loss}_{gradient_update}_bestconfig'

# Creating a folder for the current tuning
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name, exist_ok=True)


def attack_evaluate(images, labels, i, optimizer_config=None, scheduler_config=None):
    global current_date, formatted_date, experiment_name

    if optimizer_config is not None:
        print(f"Best optimizer params: {optimizer_config}")

    if scheduler_config is not None:
        print(f"Best scheduler params: {scheduler_config}")

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
        scheduler_config=scheduler_config,
        verbose=False
    )

    best_adv, _ = attack.forward(images=images, labels=labels)

    logits = model(best_adv)
    acc = (logits.cpu().argmax(dim=1) == labels.cpu()).sum().item() / batch_size
    asr = 1 - acc
    print(f"ASR check: {asr}")

    print("\t\t[FMN] Saving the attack data...")
    attack_data_path = os.path.join(experiment_name, f"{formatted_date}_attackdata_{experiment_name}_batch{i}.pth")
    torch.save(attack.attack_data, attack_data_path)


# Loading model and dataset
print("\t[FMN] Retrieving the model and the dataset...")
model, dataset, model_name, dataset_name = load_data(model_id=model_id)
model.eval()

if not parallelize:
    model = model.to(device)
else:
    model = torch.nn.DataParallel(model).to(device)

subset_indices = list(range(128*32, 128*32 + batch_size*n_batches))

print(f"Samples: {len(subset_indices)}")

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle
)

optimizer_config = None
scheduler_config = None

# Load the best params path (if not None)
if best_params_path is not None:
    opt_params = OptimizerParams[optimizer]
    sch_params = None
    if scheduler is not None:
        sch_params = SchedulerParams[scheduler]

    print(f"Loading best params")
    best_params = torch.load(best_params_path)

    optimizer_config = {k: best_params[k] for k in set(opt_params)}
    scheduler_config = None
    if scheduler is not None:
        scheduler_config = {k: best_params[k] for k in set(sch_params)}
        if 'T_max' in scheduler_config:
            scheduler_config['T_max'] = steps
        if 'batch_size' in scheduler_config:
            scheduler_config['batch_size'] = batch_size

print("\t[FMN] Starting the attack...")
for i, data in enumerate(dataloader):
    print(f"\t[FMN] Running batch {i}")

    images, labels = data
    attack_evaluate(images, labels, i, optimizer_config, scheduler_config)

    if i >= n_batches-1:
        break

print("\t[FMN] All batches completed")
