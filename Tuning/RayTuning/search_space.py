from ray import tune


OPTIMIZERS_SEARCH_TUNE = {
    'SGD': {
        'lr': tune.loguniform(8/255*2, 10),
        'momentum': tune.uniform(0.81, 0.99),
        'weight_decay': tune.loguniform(0.01, 1),
        'dampening': tune.uniform(0, 0.2)
    },

    'Adam':
    {
        'lr': tune.loguniform(8/255*2, 10),
        'weight_decay': tune.loguniform(0.01, 1),
        'eps': 1e-8,
        'beta0': tune.uniform(0.0, 1.0),
        'beta1': tune.uniform(0.0, 1.0)
    },

    'Adamax':{
        'lr': tune.loguniform(8/255*2, 10),
        'weight_decay': tune.loguniform(0.01, 1),
        'eps': 1e-8,
        'betas': tune.sample_from(lambda _: (tune.uniform(0.0, 1.0), tune.uniform(0.0, 1.0)))
    }
}

SCHEDULERS_SEARCH_TUNE = {
    'CALR':
        {
            'T_max': lambda steps: steps,
            'eta_min': 0,
            'last_epoch': -1
        },

    'RLROP':
        {
            'factor': tune.uniform(0.1, 0.5),
            'patience': tune.choice([5, 10, 20]),
            'threshold': 1e-5
        },

    'None': None
}

