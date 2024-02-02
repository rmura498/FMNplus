import os, threading
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from ax.service.ax_client import AxClient


def extact_data(filename=None):
    if filename is None:
        raise ValueError('filename is required')

    groups = filename.split('_')
    date, mid, batch_size, steps, trials, opt, sch, loss, _ = groups

    return date, mid, batch_size, steps, trials, opt, sch, loss


def save_best_params(model_id, PATH):
    json_files = tuple(Path(PATH).rglob(f'*mid{model_id}*.json'))
    print(list(json_files))

    mid_path = os.path.join('./Tuning/AxTuning/BestParams', f'mid{model_id}')
    if not os.path.exists(mid_path):
        os.makedirs(mid_path, exist_ok=True)

    for i, json_file in enumerate(json_files):
        date, mid, batch_size, steps, trials, opt, sch, loss = extact_data(json_file.name)
        conf_name = f'{opt}-{sch}-{loss}'
        best_params_name = os.path.join(mid_path, f'{conf_name}.pth')

        ax_client = AxClient(verbose_logging=False).load_from_json_file(filepath=json_file)
        ax_client.get_next_trial()

        print(conf_name)
        print(ax_client.get_best_trial())

        best_params = list(ax_client.get_best_parameters())[0]
        torch.save(best_params, best_params_name)


if __name__ == '__main__':
    for model_id in range(0, 9):
        x = threading.Thread(target=save_best_params, args=(model_id, './Tuning/AxTuning/exps'))
        x.start()
