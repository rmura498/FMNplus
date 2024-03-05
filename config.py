EXP_DIRECTORY = './Experiments'

MODEL_DATASET = {
    0: {
        'model_name': 'Wang2023Better_WRN-70-16',
        'datasets': ['cifar10']
        },
    1: {'model_name': 'Wang2023Better_WRN-28-10',
        'datasets': ['cifar10']
        },
    2: {'model_name': 'Gowal2021Improving_70_16_ddpm_100m',
        'datasets': ['cifar10']
        },
    3: {'model_name': 'Rebuffi2021Fixing_106_16_cutmix_ddpm',
        'datasets': ['cifar10']
        },
    4: {'model_name': 'Gowal2021Improving_28_10_ddpm_100m',
        'datasets': ['cifar10']
        },
    5: {'model_name': 'Pang2022Robustness_WRN70_16',
        'datasets': ['cifar10']
        },
    6: {'model_name': 'Sehwag2021Proxy_ResNest152',
        'datasets': ['cifar10']
        },
    7: {'model_name': 'Pang2022Robustness_WRN28_10',
        'datasets': ['cifar10']
        },
    8: {'model_name': 'Gowal2021Improving_R18_ddpm_100m',
        'datasets': ['cifar10']
        },
    9: {'model_name': 'Rade2021Helper_R18_ddpm',
        'datasets': ['cifar10']
        },
    10: {'model_name': 'Sehwag2021Proxy_R18',
        'datasets': ['cifar10']
        },
    11: {'model_name': 'Rebuffi2021Fixing_R18_ddpm',
        'datasets': ['cifar10']
        },
    12: {'model_name':'Liu2023Comprehensive_Swin-L',
        'datasets': ['imagenet']
        },
    13: {'model_name':'Liu2023Comprehensive_ConvNeXt-L',
        'datasets': ['imagenet']
        },
    14: {'model_name':'Singh2023Revisiting_ConvNeXt-L-ConvStem',
        'datasets': ['imagenet']
        },
    15: {
        'model_name':'Salman2020Do_R18',
        'datasets': ['imagenet']
    }
    
}

MODEL_NORMS = ["L0", "L1", "L2", "Linf"]
