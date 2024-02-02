from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour_plotly
from pathlib import Path


def extract_data(filename=None):
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
        date, mid, batch_size, steps, trials, opt, sch, loss = extract_data(json_file.name)
        conf_name = f'{opt}-{sch}-{loss}'

        # read json file
        # filepath = PATH + '/' + json_file.name
        ax_client = AxClient(verbose_logging=False).load_from_json_file(filepath=json_file)
        ax_client.get_next_trial()
        model = ax_client.generation_strategy.model
        param_y = 'beta1' if 'Adam' in opt else 'momentum'
        fig = plot_contour_plotly(model, param_x='lr', param_y=param_y, metric_name='distance', lower_is_better=True)
        fig.write_image(f'./mid_{model_id}_{conf_name}.pdf')

tun = tuning_comparison(model_id=0,PATH='tuning_results/ax_tuning_jan24/mid0' )
