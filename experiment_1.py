# In this file we model the RUL using the modified Deep Koopman approach.
# The state space vector and the control vector are encoded by different MLPs.
# The results are concatenated and passed to the linear Koopman operator.
# The output of the Koopman operator are decoded to the state space vector.


import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np
import os
from torch import Tensor
from torch.utils.data import DataLoader
from models import DeepKoopmanControl
from config import *
from utils import *
from datasets import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler


cnc_dataset = CNC_dataset(data_path, machine=machine, columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
init_data = cnc_dataset.get_data()
cnc_dataset2 = CNC_dataset(f'../DATASETS/cnc_milling_machine/c{6}', machine=6, columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
init_data2 = cnc_dataset2.get_data()

preprocessor = DESPAWN_preprocessor(despawn_params)
processed_data = preprocessor.fit_transform(init_data[['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz']])
processed_data2 = preprocessor.transform(init_data2[['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz']])
processed_data = pd.concat((processed_data, processed_data2)).reset_index(drop=True)

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
processed_data[processed_data.columns] = std_scaler.fit_transform(processed_data)
rul_train = pd.concat((init_data[['RUL']].iloc[::2, :], init_data2[['RUL']].iloc[::2, :]))
processed_data[['RUL']] = minmax_scaler.fit_transform(rul_train)

print('train_data shape', processed_data.shape)
# processed_data = processed_data.iloc[::2, :].reset_index(drop=True)

processed_data.to_feather(f'../DATASETS/cnc_milling_machine/c4/processed_data.feather')
joblib.dump(std_scaler, 'models/std_scaler.save')
joblib.dump(minmax_scaler, 'models/minmax_scaler.save')
joblib.dump(DESPAWN_preprocessor, 'models/preprocessor.save')
# processed_data = pd.read_feather(f'../DATASETS/cnc_milling_machine/c4/processed_data.feather')
# preprocessor = joblib.load('models/preprocessor.save')
# std_scaler = joblib.load('models/std_scaler.save')
# minmax_scaler = joblib.load('models/minmax_scaler.save')

x0 = Tensor(processed_data.iloc[:-1:1000, 9:18].values)
x1 = Tensor(processed_data.iloc[1::1000,9:18].values)
u0 = Tensor(processed_data.iloc[:-1:1000, :9].values)
rul1 = Tensor(processed_data.iloc[1::1000, -1:].values)

dataset = TensorDataset(x0, x1, u0, rul1)
training_loader = DataLoader(dataset, batch_size=control_model_params['bs'], shuffle=True, drop_last=True)


print('Initializing a model')
model = DeepKoopmanControl(control_model_params)
optimizer = torch.optim.Adam(model.parameters(), lr=control_model_params['lr'], weight_decay=1e-7)


print('Training the model')
for epoch in range(1, control_model_params['epochs']+1):
    train_loss, (final_pred_loss, final_autoencoder_loss, final_supervised_loss) = train_control_model(model, optimizer, training_loader)
    print(f'Epoch {epoch}. Train loss {train_loss}')
    print(f'Pred loss {final_pred_loss}, AE loss {final_autoencoder_loss}, Supervised loss {final_supervised_loss}.')

    torch.save(model.state_dict(), f'models/control_model_epoch_{epoch}.pth')
# model.load_state_dict(torch.load('models/model_epoch_100.pth'))
# model.eval()


print('Evaluation')
model.eval()
cnc_dataset = CNC_dataset(eval_data_path, machine=eval_machine, columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
init_data = cnc_dataset.get_data()

eval_data = preprocessor.transform(init_data[['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz']])
eval_data[eval_data.columns] = std_scaler.transform(eval_data)
eval_data[['RUL']] = minmax_scaler.transform(init_data[['RUL']].iloc[::2, :])


eval_data = eval_data.iloc[::10000, :]

x0 = Tensor(eval_data.iloc[:, 9:18].values)
u0 = Tensor(eval_data.iloc[:, :9].values)
y0 = model.encoder(x0)
Bu0 = model.B(u0)

observables = torch.cat((y0, Bu0), dim=1).detach()
# observables = model.encoder(Tensor(eval_data.iloc[:, 9:18].values)).detach()

# plot the evolution of observables
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

w = 100

plt.figure(figsize=(10, 5))
for obs_i in range(control_model_params['obsdim']):
    plt.plot(eval_data.index[w-1:], moving_average(observables[:, obs_i], w), label=f'obs{obs_i}')
plt.plot(eval_data.index[w-1:], eval_data.iloc[w-1:,:]['RUL'], label='RUL')
plt.legend()
plt.xlabel('step')
plt.savefig(f'plots/control/observables_RUL.png')
plt.close()

# correlation plots of observables and the RUL
for obs_i in range(control_model_params['obsdim']):
    plt.figure(figsize=(10, 10))
    plt.scatter(moving_average(observables[:, obs_i], w), eval_data.iloc[w-1:,:]['RUL'])
    plt.xlabel(f'obs dim {obs_i}')
    plt.ylabel('RUL')
    plt.savefig(f'plots/control/scatter_RUL_vs_obs{obs_i}.png')
    plt.close()

print('Modelling in the observables space')
lr_RUL = Lasso(alpha=0.01).fit(observables, eval_data['RUL'])
RUL_hat = lr_RUL.predict(observables)

plt.figure(figsize=(10, 10))
plt.plot(np.arange(eval_data['RUL'].shape[0]), eval_data['RUL'], label='ref')
# plt.plot(np.arange(RUL_hat.shape[0]), RUL_hat, label='pred')
plt.plot(np.arange(w-1, RUL_hat.shape[0]), moving_average(RUL_hat, w), label='pred', alpha=0.5)
plt.xlabel(f'step')
plt.legend()
# plt.yscale('log')
plt.savefig(f'plots/control/RUL_modelling.png')
plt.close()


# check inner linear model
RUL_hat = model.lr(observables).detach()
RUL_hat = np.array(RUL_hat).flatten()

plt.figure(figsize=(10, 10))
plt.plot(np.arange(eval_data['RUL'].shape[0]), eval_data['RUL'], label='ref')
# plt.plot(np.arange(RUL_hat.shape[0]), RUL_hat, label='pred')
plt.plot(np.arange(w-1, RUL_hat.shape[0]), moving_average(RUL_hat, w), label='pred', alpha=0.5)
plt.xlabel(f'step')
plt.legend()
# plt.yscale('log')
plt.savefig(f'plots/control/RUL_modelling_inner_model.png')
plt.close()