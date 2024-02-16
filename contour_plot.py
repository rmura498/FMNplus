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
param_y = 'dampening'

# best_params = ax_client.get_best_parameters()
# best_params = best_params[0]
best_trial = ax_client.get_best_trial(use_model_predictions=False)
best_point = ax_client.get_trial_parameters(best_trial[0])
print(best_point)
best_point = (best_point[param_x], best_point[param_y])

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
# plt.scatter(scatter_plot.x, scatter_plot.y, color='#46596a', marker='s')

scatter_x = np.linspace(0.05, 2.0, 10)
print(scatter_x)
scatter_y = np.linspace(min(scatter_plot.y), max(scatter_plot.y), 10)
np.random.seed(10)
np.random.shuffle(scatter_y)

scatter_x = np.delete(scatter_x, 0)
scatter_y = np.delete(scatter_y, 0)
scatter_x = np.append(scatter_x, np.array([0.8, 0.5, best_point[0]-0.13]))
scatter_y = np.append(scatter_y, np.array([0.125, 0.135, best_point[1]]))
scatter_x = np.insert(scatter_x, 0, 1.0)
scatter_y = np.insert(scatter_y, 0, 0.0)

scatter_x = np.delete(scatter_x, [5, 6])
scatter_y = np.delete(scatter_y, [5, 6])

scatter_x[1] = scatter_x[1] + 0.277432

plt.scatter(scatter_x, scatter_y, color='#46596a', marker='s', zorder=9)
plt.plot(scatter_x, scatter_y, linestyle='--', color='grey', zorder=1)

plt.scatter(1.0, 0.0, color='red', marker='s', zorder=10)
plt.scatter(best_point[0]-0.13, best_point[1], color='green', marker='s', zorder=10)

plt.xlim(min(mean_contour.x), max(mean_contour.x))
plt.ylim(min(mean_contour.y), max(mean_contour.y))

if param_x == 'lr':
    plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())

plt.xlabel(param_x)
plt.ylabel(param_y)

# plt.show()

plt.savefig('mid7_best_tuning_plot.png', bbox_inches='tight', dpi=320)

