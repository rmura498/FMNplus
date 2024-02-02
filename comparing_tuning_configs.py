import threading
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from ax.service.ax_client import AxClient

columns_to_remove = {'eps', 'T_max', 'eta_min', 'last_epoch', 'batch_size', 'threshold'}


def extact_data(filename=None):
    if filename is None:
        raise ValueError('filename is required')

    groups = filename.split('_')
    date, mid, batch_size, steps, trials, opt, sch, loss, _ = groups

    return date, mid, batch_size, steps, trials, opt, sch, loss


def tuning_comparison(model_id, PATH):
    configs_dict = {}
    json_files = tuple(Path(PATH).rglob(f'*mid{model_id}*.json'))
    print(list(json_files))

    for i, json_file in enumerate(json_files):
        date, mid, batch_size, steps, trials, opt, sch, loss = extact_data(json_file.name)
        conf_name = f'{opt}-{sch}-{loss}'

        # read json file
        # filepath = PATH + '/' + json_file.name
        ax_client = AxClient(verbose_logging=False).load_from_json_file(filepath=json_file)

        print(conf_name)
        ax_client.get_next_trial()

        conv_dict = {
            'lr': lambda x:f'{x:.4e}',
            'momentum':lambda x:f'{x:.4f}',
            'weight_decay':lambda x:f'{x:.3f}',
            'dampening':lambda x:f'{x:.3f}',
            'beta1':lambda x:f'{x:.3f}',
            'beta2':lambda x:f'{x:.3f}',
            'factor':lambda x:f'{x:.3f}'
        }
        print(ax_client.get_best_trial())
        best_params = ax_client.get_best_parameters()
        best_params = list(best_params)
        best_params_dict = best_params[0]
        #print(best_params_dict)
        for k, v in best_params_dict.items():
            if k in conv_dict:
                best_params_dict[k] = conv_dict[k](v)

        #print(best_params_dict)
        best_results = best_params[1][0]

        configs_dict[conf_name] = {**(best_params_dict)}
        configs_dict[conf_name]['Best Distance'] = best_results['distance']

        if 'beta1' in configs_dict[conf_name]:
            betas = f"({configs_dict[conf_name]['beta1']},{configs_dict[conf_name]['beta2']})"
            configs_dict[conf_name]['betas'] = betas
            del configs_dict[conf_name]['beta1']
            del configs_dict[conf_name]['beta2']

    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(configs_dict)
    df = df.transpose()
    df = df.sort_values(by=['Best Distance'], ascending=True)

    # remove NaNs
    df.fillna(value='-', inplace=True)
    last_col = len(df.columns)
    column_to_move = df.pop("Best Distance")
    df.insert(last_col-1, 'Best Distance', column_to_move)

    for column in df.columns:
        if column in columns_to_remove:
            df.drop(column, axis=1, inplace=True)

    print(df)

    from datetime import datetime
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y%H")
    df.to_latex(f'tuning_results_mid{model_id}_{formatted_date}.tex')


def trial_comparison(PATH):

    configs_dict = {}
    json_files = Path(PATH).rglob('*.json')

    nrows = 2
    ncols = 3
    figure, axes = plt.subplots(nrows, ncols, figsize=(12, 8))

    for i, json_file in enumerate(json_files):
        date, mid, batch_size, steps, trials, opt, sch, loss = extact_data(json_file.name)
        conf_name = f'{opt}-{sch}-{loss}'

        filepath = PATH + '/' + json_file.name
        ax_client = AxClient().load_from_json_file(filepath=filepath, verbose_logging=False)
        ax_client.get_next_trial()

        print(ax_client.get_trace())
        #print(ax_client.get_trace_by_progression())


        median_distances = ax_client.get_trace()
        x = np.linspace(0, len(median_distances), len(median_distances))
        median_distances = torch.tensor(median_distances)

        median_value = median_distances.median().item()

        ax = axes.flatten()[i]

        ax.plot(x, median_distances, color='#3D5A80')
        ax.set_title(conf_name+f'-Best-{median_distances[-1]:.5f}')
        ax.grid(True)

        plt.ylim(0, 0.07)
        ax.legend()

    figure.tight_layout(pad=1.5)
    figure.text(0.5, 0.01, r'Trials', ha='center', fontsize='large')
    figure.text(0.001, 0.5, 'Median', va='center', rotation='vertical', fontsize='large')

    plt.savefig(f"comparison_trials.pdf", bbox_inches='tight', dpi=320)


if __name__ == '__main__':
    for model_id in range(0, 9):
        x = threading.Thread(target=tuning_comparison, args=(model_id, './Tuning/AxTuning/exps'))
        x.start()
        # tuning_comparison(model_id=model_id, PATH='./Tuning/AxTuning/exps')

    # tuning_comparison(model_id=3, PATH='./Tuning/AxTuning/exps/mid8')
