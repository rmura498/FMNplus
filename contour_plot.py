import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour_plotly

path = r"Tuning/AxTuning/exps/mid7_128_200_32_SGD_CALR_DLR_Sign/31012406_mid7_128_200_32_SGD_CALR_DLR_Sign.json"
ax_client = AxClient(verbose_logging=False).load_from_json_file(filepath=path)
model = ax_client.generation_strategy._curr.model(
    experiment=ax_client.experiment,
    data=ax_client.experiment.lookup_data(),
    **ax_client.generation_strategy._curr.model_kwargs
)
ax_client.get_next_trial()

# import ipdb; ipdb.set_trace()

param_x = 'lr'
param_y = 'momentum'

best_params = ax_client.get_best_parameters()
best_params = best_params[0]
best_point = (best_params[param_x], best_params[param_y])

print(f"Best point: {best_point}")

# plotly returns a Figure object, you can access plots' data through data.data (kinda the list of plots' data)
data = plot_contour_plotly(
    model=model,
    param_x=param_x,
    param_y=param_y,
    metric_name='distance',
    lower_is_better=True)

plots_data = data.data
mean_contour = plots_data[0]
scatter_plot = data.data[-1]

plt.contourf(mean_contour.x, mean_contour.y, mean_contour.z, levels=15, cmap='Blues')
plt.contour(mean_contour.x, mean_contour.y, mean_contour.z, levels=15, colors='black', linewidths=0.5)
plt.scatter(scatter_plot.x, scatter_plot.y, color='#46596a', marker='s')

plt.scatter(1.0, 0.0, color='red', marker='s')
plt.scatter(best_point[0], best_point[1], color='green', marker='s', zorder=10)

plt.xlim(8/255, 10)
plt.ylim(0.0, 0.9)
plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())

plt.xlabel(param_x)
plt.ylabel(param_y)

plt.show()

