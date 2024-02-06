from ray import tune


OptimizerParams = {
    'SGD': {
        'lr': tune.loguniform(4/255, 10),
        'momentum': tune.uniform(0.0, 0.9),
        'weight_decay': tune.loguniform(0.01, 1),
        'dampening': tune.uniform(0, 0.2)
    },

    'Adam':
    {
        'lr': tune.loguniform(4/255*2, 10),
        'weight_decay': tune.loguniform(0.01, 1),
        'eps': 1e-8,
        'beta1': tune.uniform(0.0, 1.0),
        'beta2': tune.uniform(0.0, 1.0)
    },

    'Adamax':{
        'lr': tune.loguniform(8/255*2, 10),
        'weight_decay': tune.loguniform(0.01, 1),
        'eps': 1e-8,
        'beta1': tune.uniform(0.0, 1.0),
        'beta2': tune.uniform(0.0, 1.0)
    }
}

SchedulerParams = {
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
        }
}

