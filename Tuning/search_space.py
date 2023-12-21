from ray import tune


OPTIMIZERS_SEARCH_TUNE = {
    'SGD': {
        'lr': tune.loguniform(1, 100),
        'momentum': tune.uniform(0.81, 0.99),
        'weight_decay': tune.loguniform(0.01, 1),
        'dampening': tune.uniform(0, 0.2)
    },

    'Adam':
    {
        'lr': tune.loguniform(1, 100),
        'weight_decay': tune.loguniform(0.01, 1),
        'eps': 1e-8,
        'betas': (0.9, 0.999)
    },

    'Adamax':{
        'lr': tune.loguniform(1, 100),
        'weight_decay': tune.loguniform(0.01, 1),
        'eps': 1e-8,
        'betas': (0.9, 0.999)
    }
}

SCHEDULERS_SEARCH_TUNE = {
    'CosineAnnealingLR':
        {
            'T_max': lambda steps: steps,
            'eta_min': 0,
            'last_epoch': -1
        },

    'ReduceLROnPlateau':
        {
            'factor': tune.uniform(0.1, 0.5),
            'patience': tune.choice([5, 10, 20]),
            'threshold': 1e-5
        }
}

