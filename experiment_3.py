# In this file we model the RUL using the modified Deep Koopman approach.
# The state space vector is encoded and decoded by MLP.
# The RUL is modelled linearly in the observables space.


import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import os
from torch import Tensor
from torch.utils.data import DataLoader
from models import DeepKoopmanExperiment3
from config import *
from utils import *
from datasets import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pickle
import gc
import tensorflow as tf
import tensorflow.keras as keras
import json


data_processed = True
train_machines = [(1, 4), (1, 6), (4, 6)]
test_machines = [6, 4, 1]
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

rmse = {}
mean_rul_prediction_error = {}
rul_prediction_error = {}

for (train_machine1, train_machine2), test_machine in zip(train_machines, test_machines):
    if data_processed:
        processed_data = pd.read_feather(f'../DATASETS/cnc_milling_machine/processed_data_machines_{train_machine1}{train_machine2}.feather')
        std_scaler = joblib.load(f'models/experiment_1/std_scaler_{train_machine1}{train_machine2}.save')
        minmax_scaler = joblib.load(f'models/experiment_1/minmax_scaler_{train_machine1}{train_machine2}.save')
    else:
        train_dataset_1 = CNC_dataset(f'../DATASETS/cnc_milling_machine/c{train_machine1}', machine=train_machine1, columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
        train_data_1 = train_dataset_1.get_data()#.iloc[:100000, :]
        train_dataset_2 = CNC_dataset(f'../DATASETS/cnc_milling_machine/c{train_machine2}', machine=train_machine2, columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
        train_data_2 = train_dataset_2.get_data()#.iloc[:100000, :]

        preprocessor = DESPAWN_preprocessor(despawn_params)
        train_processed_1 = preprocessor.fit_transform(train_data_1[['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz']])
        train_processed_2 = preprocessor.transform(train_data_2[['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz']])
        processed_data = pd.concat((train_processed_1, train_processed_2)).reset_index(drop=True)

        std_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        processed_data[processed_data.columns] = std_scaler.fit_transform(processed_data)
        rul_train = pd.concat((train_data_1[['RUL']].iloc[::2, :], train_data_2[['RUL']].iloc[::2, :]))
        processed_data[['RUL']] = minmax_scaler.fit_transform(rul_train)

        processed_data.to_feather(f'../DATASETS/cnc_milling_machine/processed_data_machines_{train_machine1}{train_machine2}.feather')
        joblib.dump(std_scaler, f'models/experiment_3/std_scaler_{train_machine1}{train_machine2}.save')
        joblib.dump(minmax_scaler, f'models/experiment_3/minmax_scaler_{train_machine1}{train_machine2}.save')

    x0 = Tensor(processed_data.iloc[::1000, :18].values)
    x1 = Tensor(processed_data.iloc[1::1000, :18].values)
    rul = Tensor(processed_data.iloc[::1000, -1:].values)

    dataset = TensorDataset(x0, x1, rul)
    training_loader = DataLoader(dataset, batch_size=model_params_exp3['bs'], shuffle=True, drop_last=True)


    print('Initializing a model')
    model = DeepKoopmanExperiment3(model_params_exp3)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params_exp3['lr'], weight_decay=1e-7)


    print('Training the model')
    for epoch in range(1, model_params_exp3['epochs']+1):
        train_loss, (supervised_loss, prediction_loss, reconstruction_loss) = train_model_exp3(model, optimizer, training_loader)
        print(f'Epoch {epoch}. Train loss {train_loss}')
        print(f'Supervised loss {supervised_loss}, Prediction loss {prediction_loss}, AE loss {reconstruction_loss}.')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'models/experiment_2/model_epoch_{epoch}.pth')
    # model.load_state_dict(torch.load('models/model_epoch_100.pth'))
    # model.eval()

    try:
        del train_dataset_1, train_data_1, train_dataset_2, train_data_2, processed_data
        gc.collect()
    except:
        pass

    print('Evaluation')
    model.eval()
    if data_processed:
        test_processed = pd.read_feather(f'../DATASETS/cnc_milling_machine/processed_data_machines_{test_machine}.feather')
    else:
        test_dataset = CNC_dataset(f'../DATASETS/cnc_milling_machine/c{test_machine}', machine=test_machine,
                                   columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
        test_data = test_dataset.get_data()
        test_processed = preprocessor.transform(test_data[['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz']])
        test_processed[test_processed.columns] = std_scaler.transform(test_processed)
        test_processed[['RUL']] = minmax_scaler.transform(test_data[['RUL']].iloc[::2, :])
        test_processed.to_feather(f'../DATASETS/cnc_milling_machine/processed_data_machines_{test_machine}.feather')
    test_processed = test_processed.iloc[::1000, :]

    x = Tensor(test_processed.iloc[:, :18].values)
    observables = model.encoder(x).detach()


    # plot the evolution of observables
    w = 1000

    plt.figure(figsize=(10, 5))
    for obs_i in range(model_params_exp3['obsdim']):
        plt.plot(test_processed.index[w - 1:], moving_average(observables[:, obs_i], w), label=f'obs{obs_i}')
    plt.plot(test_processed.index[w - 1:], test_processed.iloc[w - 1:, :]['RUL'], label='RUL')
    plt.legend()
    plt.xlabel('step')
    plt.savefig(f'plots/experiment_3/observables_RUL_machine_{test_machine}.png')
    plt.close()

    # check inner linear model
    RUL_hat = model.lr(observables).detach()
    RUL_hat = np.array(RUL_hat).flatten()

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(test_processed['RUL'].shape[0]), test_processed['RUL'], label='Reference RUL')
    plt.plot(np.arange(w-1, RUL_hat.shape[0]), moving_average(RUL_hat, w), label='DK', alpha=0.5)
    plt.xlabel(f'step')
    plt.legend()
    # plt.yscale('log')
    plt.savefig(f'plots/experiment_3/RUL_modelling_inner_model_machine_{test_machine}.png')
    plt.close()

    print('MSE', mean_squared_error(test_processed['RUL'], RUL_hat))
    print('MAE', mean_absolute_error(test_processed['RUL'], RUL_hat))
    print('MAPE', mean_absolute_percentage_error(test_processed['RUL'], RUL_hat))

    # metrics calculation
    rmse[f'machine_{test_machine}'] = mean_squared_error(test_processed['RUL'], RUL_hat)

    window_size = 1000
    moving_error = custom_rul_metric(RUL_hat, w=window_size)
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(0, len(test_processed['RUL']), window_size), moving_error, label='RUL prediction error')
    plt.xlabel(f'step')
    plt.legend()
    # plt.yscale('log')
    plt.savefig(f'plots/experiment_3/custom_metric_machine_{test_machine}.png')
    plt.close()

    mean_rul_prediction_error[f'machine_{test_machine}'] = np.mean(moving_error)
    rul_prediction_error[f'machine_{test_machine}'] = moving_error

    try:
        del preprocessor, test_dataset, test_data, train_dataset_1, train_dataset_2
    except:
        pass
    del dataset, training_loader, test_processed, observables
    gc.collect()

rmse = pd.DataFrame.from_dict(rmse, orient='index')
rmse.to_csv('models/experiment_3/rmse.csv')

joblib.dump(mean_rul_prediction_error, 'models/experiment_3/mean_custom_metric.txt')
joblib.dump(rul_prediction_error, 'models/experiment_3/custom_metric.txt')