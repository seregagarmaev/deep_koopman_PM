import joblib
import numpy as np
import pandas as pd

for experiment in [1, 2, 3]:
    rmse = pd.read_csv(f'models/experiment_{experiment}/rmse.csv')
    print(f'RMSE. Experiment {experiment}.')
    print(rmse)
    print('Mean RMSE:', rmse.mean()[0])

    mean_metric = joblib.load(f'models/experiment_{experiment}/mean_custom_metric.txt')
    print(f'Mean RUL prediction metric. Experiment {experiment}.')
    print(mean_metric)
    print('Averaged over machines RUL prediction metric', np.mean([*mean_metric.values()]))