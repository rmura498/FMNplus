from pathlib import Path
import numpy as np
import pandas as pd

from ax.service.ax_client import AxClient


def extact_data(filename=None):
    if filename is None:
        raise ValueError('filename is required')

    groups = filename.split('_')
    date, mid, batch_size, steps, trials, opt, sch, loss, _ = groups

    return date, mid, batch_size, steps, trials, opt, sch, loss


def tuning_comparison(model_id, PATH):
    configs_dict = {}
    json_files = Path(PATH).rglob('*.json')

    for json_file in json_files:
        date, mid, batch_size, steps, trials, opt, sch, loss = extact_data(json_file.name)
        conf_name = f'{opt}-{sch}-{loss}'

        # read json file
        filepath = PATH + '/' + json_file.name
        ax_client = AxClient().load_from_json_file(filepath=filepath, verbose_logging=False)

        ax_client.get_next_trial()

        best_params = ax_client.get_pareto_optimal_parameters()
        best_params = best_params.items()[0]
        print(best_params)
        best_params, values = best_params
        print(best_params, values)
        exit(0)

        check_mid = configs.get(mid, None)
        check_conf = None

        if check_mid is not None:
            check_conf = configs[mid].get(conf_name, None)
        else:
            configs[mid] = {}

        if check_conf is None:
            configs[mid][conf_name] = [sr_data, ]
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

    sr_df = sr_df.style.highlight_max(
        props='cellcolor:[HTML]{b666cc}; color:{white};' 'textit:--rwrap; textbf:--rwrap;').format("{:.2f}")

    from datetime import datetime
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y%H")
    sr_df.to_latex(f'sr_comparison_{formatted_date}.txt')


tuning_comparison(model_id=0, PATH='tuning_results/ax_tuning_jan24/mid0')
