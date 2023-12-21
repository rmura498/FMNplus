import pickle, io
from pathlib import Path

import numpy
import torch
import numpy as np
import pandas as pd

PATH = Path("./Experiments")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def extact_data(filename=None):
    if filename is None:
        raise ValueError('filename is required')

    groups = filename.split('_')
    date, mid, attack, eps, batch_size, steps, opt, sch, loss, grad, ei, bid = groups
    bid = bid.replace('.pkl', '')

    return date, mid, attack, eps, batch_size, steps, opt, sch, loss, grad, ei, bid


configs = {}

pkl_files = Path(PATH).rglob('*.pkl')
for pkl_file in pkl_files:
    print(pkl_file.name)

    date, mid, attack, eps, batch_size, steps, opt, sch, loss, grad, ei, bid = extact_data(pkl_file.name)

    if attack != 'AA':
        conf_name = f'{attack}-{opt}-{sch}-{loss}'
    else:
        conf_name = f'{attack}-{loss}'

    # read pkl file
    with pkl_file.open('rb') as f:
        # picking only last lr
        sr_data = CPU_Unpickler(f).load()['success_rate'][-1]

        if torch.is_tensor(sr_data):
            sr_data = sr_data.item()

    check_mid = configs.get(mid, None)
    check_conf = None

    if check_mid is not None:
        check_conf = configs[mid].get(conf_name, None)
    else:
        configs[mid] = {}

    if check_conf is None:
        configs[mid][conf_name] = [sr_data,]
    else:
        configs[mid][conf_name].append(sr_data)

sr_df = {}
for mid in configs:
    for config in configs[mid]:
        configs[mid][config] = np.array(configs[mid][config]).mean().round(decimals=2)

sr_df = pd.DataFrame(configs)
sr_df['SR-mean'] = sr_df.mean(axis=1)
sr_df = sr_df.sort_values('SR-mean', ascending=False)

print(sr_df)

sr_df = sr_df.style.highlight_max(props='cellcolor:[HTML]{b666cc}; color:{white};' 'textit:--rwrap; textbf:--rwrap;').format("{:.2f}")

from datetime import datetime
current_date = datetime.now()
formatted_date = current_date.strftime("%d%m%y%H")
sr_df.to_latex(f'sr_comparison_{formatted_date}.txt')

