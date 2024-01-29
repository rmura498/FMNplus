from pathlib import Path
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

    for i, json_file in enumerate(json_files):
        date, mid, batch_size, steps, trials, opt, sch, loss = extact_data(json_file.name)
        conf_name = f'{opt}-{sch}-{loss}'

        # read json file
        filepath = PATH + '/' + json_file.name
        ax_client = AxClient().load_from_json_file(filepath=filepath, verbose_logging=False)

        print(conf_name)
        ax_client.get_next_trial()

        best_params = ax_client.get_best_parameters()

        best_params = list(best_params)
        best_params_dict = best_params[0]
        best_results = best_params[1][0]

        configs_dict[conf_name] = {'Best Params':best_params_dict,
                                   'Best Distance':best_results['distance']}
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(configs_dict)
    df = df.transpose()
    df = df.sort_values(by=['Best Distance'], ascending=True)
    print(df)
    from datetime import datetime
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y%H")
    df.to_latex(f'sr_comparison-mid{model_id}_{formatted_date}.txt')


tuning_comparison(model_id=8, PATH='tuning_results/ax_tuning_jan24/mid8_median')
