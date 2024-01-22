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

    for json_file in json_files:
        date, mid, batch_size, steps, trials, opt, sch, loss = extact_data(json_file.name)
        conf_name = f'{opt}-{sch}-{loss}'

        # read json file
        filepath = PATH + '/' + json_file.name
        ax_client = AxClient().load_from_json_file(filepath=filepath, verbose_logging=False)

        ax_client.get_next_trial()

        best_params = ax_client.get_pareto_optimal_parameters()
        best_params = list(best_params.items())
        best_params_dict = best_params[0][1][0]
        best_results = best_params[0][1][1][0]

        configs_dict[conf_name] = {'Best Params':str(best_params_dict),
                                   'Best Distance':best_results['distance'],
                                   'SR':best_results['sr']}
    print(configs_dict)
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(configs_dict)
    print(df)
    df = df.transpose()
    df = df.sort_values(by=['SR'], ascending=False)
    print(df)
    from datetime import datetime
    current_date = datetime.now()
    formatted_date = current_date.strftime("%d%m%y%H")
    df.to_latex(f'sr_comparison_{formatted_date}.txt')


tuning_comparison(model_id=0, PATH='tuning_results/ax_tuning_jan24/mid0')
