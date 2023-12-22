import os
import pickle

import torch
from torch.utils.data import DataLoader
from torch import inf

import ray
from ray import tune, train, air
from ray.air import session
from ray.tune.search.ax import AxSearch

from Utils.load_model import load_data

from Tuning.fmn_HO import HOFMN
from Tuning.search_space import OPTIMIZERS_SEARCH_TUNE, SCHEDULERS_SEARCH_TUNE
from Tuning.tuning_resources import TUNING_RES


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ray.init()


def tune_attack(config, model, dataloader, attack_params, tuning_metric_name, report_inside=True, epochs=5):

    # for epoch in range(epochs):
    scheduler_config = config.get('sch_s', None)
    attack = HOFMN(
        model=model,
        norm=attack_params['norm'],
        steps=attack_params['steps'],
        optimizer=attack_params['optimizer'],
        scheduler=attack_params['scheduler'],
        optimizer_config=config['opt_s'],
        scheduler_config=scheduler_config,
        epsilon=attack_params['epsilon'],
        device=device
    )

    inputs, labels = next(iter(dataloader))
    inputs, labels = inputs.to(device), labels.to(device)

    tuning_metric, _ = attack.forward(
        images=inputs,
        labels=labels,
        epochs=epochs,
        tuning_metric_name=tuning_metric_name,
        report_inside=report_inside
    )

    if not report_inside:
        train.report(dict(tuning_metric))
        # LS: or ... as it was before, but was causing problems (maybe ?)
        # session.report(tuning_metric)


def tune_fmn(
        model,
        dataloader,
        optimizer,
        scheduler,
        batch,
        steps,
        num_samples,
        loss,
        epochs=1,
        epsilon=None,
        tuning_metric_name=None,
        report_inside=True
):
    tuning_metric_name = 'distance' if tuning_metric_name is None else tuning_metric_name

    # load search spaces
    optimizer_search = OPTIMIZERS_SEARCH_TUNE[optimizer]
    scheduler_search = SCHEDULERS_SEARCH_TUNE[scheduler]

    attack_params = {
        'batch': int(batch),
        'steps': int(steps),
        'norm': inf,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss': loss,
        'epsilon': epsilon
    }

    tune_config = {
        'num_samples': int(num_samples),
        'epochs': int(epochs)
    }

    if scheduler_search is not None and 'T_max' in scheduler_search:
        scheduler_search['T_max'] = attack_params['steps']
        # scheduler_search['T_max'] = scheduler_search['T_max'](attack_params['steps'])

    if scheduler_search is not None:
        search_space = {
            'opt_s': optimizer_search,
            'sch_s': scheduler_search
        }
    else:
        search_space = {
            'opt_s': optimizer_search
        }

    tune_with_resources = tune.with_resources(
        tune.with_parameters(
            tune_attack,
            model=torch.nn.DataParallel(model).to(device),
            dataloader=dataloader,
            attack_params=attack_params,
            epochs=tune_config['epochs'],
            tuning_metric_name=tuning_metric_name,
            report_inside=report_inside
        ),
        resources=TUNING_RES
    )

    '''
    tune_scheduler = ASHAScheduler(mode='min', metric='distance', grace_period=2)
    algo = CFO(metric='distance', mode='min')

    tuning_exp_name = f"{optimizer}_{scheduler}_{loss}"
    tuner = tune.Tuner(
        tune_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=tune_config['num_samples'],
            search_alg=algo,
            scheduler=tune_scheduler
        ),
        run_config=air.RunConfig(
            tuning_exp_name,
            local_dir='./tuning_data',
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=False,
                checkpoint_frequency=0,
                num_to_keep=None),
            log_to_file=False,
            verbose=1
        )
    )
    '''

    algo = AxSearch()

    tuner = tune.Tuner(
        tune_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=tune_config['num_samples'],
            metric='success_rate',
            mode='max',
            search_alg=algo
        )
    )
    
    results = tuner.fit()
    ray.shutdown()

    # Checking best result and best config
    best_result = results.get_best_result(metric='success_rate', mode='max')
    best_config = best_result.config
    print(f"best config : {best_config}\n")

    return best_config


if __name__ == '__main__':
    # TODO: add argparse for the arguments defined below (still to add: cuda device etc)

    model_id = 8
    batch_size = 100
    steps = 30
    shuffle = True
    # LS: from tests defining more samples (or trials) is better for exploring more the search space
    num_samples = 20
    epsilon = 8/255
    epochs = 10

    optimizer = 'SGD'
    scheduler = 'CALR'
    loss = 'LL'

    tuning_metric_name = 'distance'
    report_inside = True

    model, dataset, model_name, dataset_name = load_data(model_id=model_id)
    model.eval()
    model = model.to(device)

    # _bs = batch_size + 250 # use this for cleaning misclassified points
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    best_config = tune_fmn(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        batch=batch_size,
        steps=steps,
        num_samples=num_samples,
        loss=loss,
        epochs=epochs,
        epsilon=epsilon,
        tuning_metric_name=tuning_metric_name,
        report_inside=report_inside
    )

    print("Saving best model config...")

    from datetime import datetime
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y%H")

    tuning_exp_name = f"{formatted_date}_{optimizer}_{scheduler}_{loss}.pkl"
    tuning_exp_path = os.path.join('tuning_data', 'best_configs')
    os.makedirs(tuning_exp_path, exist_ok=True)

    with open(os.path.join(tuning_exp_path, tuning_exp_name), 'wb') as fp:
        pickle.dump(best_config, fp)
