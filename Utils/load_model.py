import os

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from robustbench.utils import load_model as rb_load_model

from Utils.tiny_imagenet import download_dataset, unzip_data

from config import MODEL_NORMS, MODEL_DATASET


def load_dataset(dataset_name='cifar10'):
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    '''
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST('./Models/data',
                                             train=False,
                                             download=True,
                                             transform=transform)
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10('./Models/data',
                                                   train=False,
                                                   download=True,
                                                   transform=transform)
    elif dataset_name == 'imagenet':
        if not os.path.exists('./Models/data/tiny-imagenet-200'):
            download_dataset()
            unzip_data()
        dataset = datasets.ImageFolder(os.path.join('./Models/data/tiny-imagenet-200', 'test'), transform=transform)
    else:
        raise NotImplementedError("Unknown dataset")

    return dataset


def load_model(model_name, dataset_name, norm="Linf"):
    if norm not in MODEL_NORMS:
        norm = "Linf"

    try:
        model = rb_load_model(
            model_dir="./Models/pretrained",
            model_name=model_name,
            dataset=dataset_name,
            norm=norm
        )
    except KeyError:
        model = rb_load_model(
            model_dir="./Models/pretrained",
            model_name=MODEL_DATASET[0]['model_name'],
            dataset=MODEL_DATASET[0]['datasets'][0],
            norm='Linf'
        )

    return model


def load_data(model_id=0, dataset_id=0, norm='inf'):
    """
    Load model and dataset (default: Gowal2021Improving_R18_ddpm_100m, CIFAR10)
    """
    model_id=int(model_id)
    dataset_id = int(dataset_id)
    model_id = 0 if model_id > len(MODEL_DATASET) else model_id
    dataset_id = 0 if dataset_id > len(MODEL_DATASET[model_id]['datasets']) else dataset_id

    model_name = MODEL_DATASET[model_id]['model_name']
    dataset_name = MODEL_DATASET[model_id]['datasets'][dataset_id]

    model = load_model(model_name, dataset_name, norm)
    dataset = load_dataset(dataset_name)

    return model, dataset, model_name, dataset_name
