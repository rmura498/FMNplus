from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour_plotly
from pathlib import Path

import threading


def extract_data(filename=None):
    if filename is None:
        raise ValueError('filename is required')

    groups = filename.split('_')
    date, mid, batch_size, steps, trials, opt, sch, loss, _ = groups

    return date, mid, batch_size, steps, trials, opt, sch, loss


def read_json(json_file):
    date, mid, batch_size, steps, trials, opt, sch, loss = extract_data(json_file.name)
    conf_name = f'{opt}-{sch}-{loss}'

    # read json file
    # filepath = PATH + '/' + json_file.name
    ax_client = AxClient(verbose_logging=False).load_from_json_file(filepath=json_file)
    ax_client.get_next_trial()
    model = ax_client.generation_strategy.model
    param_x = 'beta1' if 'Adam' in opt else 'momentum'
    fig = plot_contour_plotly(model, param_x=param_x, param_y='lr', metric_name='distance', lower_is_better=True)
    with open(f'{mid}_{conf_name}.png', 'wb') as f:
        f.write(fig.to_image('png'))


def tuning_comparison(model_id, PATH):
    json_files = tuple(Path(PATH).rglob(f'*mid{model_id}*.json'))

    print(len(list(json_files)))

    for i, json_file in enumerate(json_files):
        x = threading.Thread(target=read_json,args=(json_file,))
        x.start()
        # read_json(json_file)


if __name__ == '__main__':
    tuning_comparison(model_id=0,PATH='Tuning/AxTuning/exps' )
