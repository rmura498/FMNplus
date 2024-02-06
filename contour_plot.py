import torch
import numpy as np
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient
from ax.plot.contour import interact_contour

path = 'tuning_results/ax_tuning_jan24/mid7/31012406_mid7_128_200_32_SGD_CALR_DLR_Sign.json'
ax_client = AxClient(verbose_logging=False).load_from_json_file(filepath=path)
model = ax_client.generation_strategy._curr.model(
    experiment=ax_client.experiment,
    data=ax_client.experiment.lookup_data(),
    **ax_client.generation_strategy._curr.model_kwargs
)
ax_client.get_next_trial()
data = list(interact_contour(model=model, metric_name='distance', lower_is_better=True))

x = data[0]['data'][0]['x'][:]
y = data[0]['data'][0]['y'][:]
z = data[0]['data'][0]['z'][:]

[X, Y] = np.meshgrid(x, y)

plt.contour(X, Y, z, levels=4)
plt.colorbar()
plt.show()
