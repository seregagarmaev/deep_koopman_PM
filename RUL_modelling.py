import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from torch import Tensor
from torch.utils.data import DataLoader
from models import DeepKoopman
from config import *
from utils import *
from datasets import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler


cnc_dataset = CNC_dataset(data_path, machine=machine, columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
init_data = cnc_dataset.get_data()
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
init_data[init_data.columns[:-1]] = std_scaler.fit_transform(init_data[init_data.columns[:-1]])
init_data[init_data.columns[-1:]] = minmax_scaler.fit_transform(init_data[init_data.columns[-1:]])

dataset = TensorDataset(Tensor(init_data.iloc[:-1:1000, :-2].values),
                        Tensor(init_data.iloc[1::1000, :-2].values),
                        Tensor(init_data.iloc[1::1000, -1:].values))
training_loader = DataLoader(dataset, batch_size=model_params['bs'], shuffle=True, drop_last=True)


print('Initializing a model')
model = DeepKoopman(model_params)
optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay=1e-7)


print('Training the model')
for epoch in range(1, model_params['epochs']+1):
    train_loss = train_model(model, optimizer, training_loader)
    print(f'Epoch {epoch}. Train loss {train_loss}')

    torch.save(model.state_dict(), f'models/model_epoch_{epoch}.pth')
# model.load_state_dict(torch.load('models/model_epoch_100.pth'))
# model.eval()


print('Evaluation')
cnc_dataset = CNC_dataset(eval_data_path, machine=eval_machine, columns=data_columns, use_RUL=True, wear_threshold=wear_threshold)
init_data = cnc_dataset.get_data()
eval_data = init_data.iloc[:-1:1000, :]

eval_data[eval_data.columns[:-1]] = std_scaler.transform(eval_data[eval_data.columns[:-1]])
eval_data[eval_data.columns[-1:]] = minmax_scaler.transform(eval_data[eval_data.columns[-1:]])

observables = model.encoder(Tensor(eval_data.iloc[:, :-2].values)).detach()

# plot the evolution of observables
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

w = 100

plt.figure(figsize=(10, 5))
for obs_i in range(model_params['obsdim']):
    plt.plot(eval_data.index[w-1:], moving_average(observables[:, obs_i], w), label=f'obs{obs_i}')
plt.plot(eval_data.index[w-1:], eval_data.iloc[w-1:,:]['RUL'], label='RUL')
plt.legend()
plt.xlabel('step')
plt.savefig(f'plots/observables_RUL.png')
plt.close()

# correlation plots of observables and the RUL
for obs_i in range(model_params['obsdim']):
    plt.figure(figsize=(10, 10))
    plt.scatter(moving_average(observables[:, obs_i], w), eval_data.iloc[w-1:,:]['RUL'])
    plt.xlabel(f'obs dim {obs_i}')
    plt.ylabel('RUL')
    plt.savefig(f'plots/scatter_RUL_vs_obs{obs_i}.png')
    plt.close()

print('Modelling in the observables space')
lr_RUL = Lasso(alpha=0.1).fit(observables, eval_data['RUL'])
RUL_hat = lr_RUL.predict(observables)

plt.figure(figsize=(10, 10))
plt.plot(np.arange(eval_data['RUL'].shape[0]), eval_data['RUL'], label='ref')
# plt.plot(np.arange(RUL_hat.shape[0]), RUL_hat, label='pred')
plt.plot(np.arange(10-1, RUL_hat.shape[0]), moving_average(RUL_hat, 10), label='pred', alpha=0.5)
plt.xlabel(f'step')
plt.legend()
# plt.yscale('log')
plt.savefig(f'plots/RUL_modelling.png')
plt.close()


# check inner linear model
RUL_hat = model.lr(observables).detach()
RUL_hat = np.array(RUL_hat).flatten()

plt.figure(figsize=(10, 10))
plt.plot(np.arange(eval_data['RUL'].shape[0]), eval_data['RUL'], label='ref')
# plt.plot(np.arange(RUL_hat.shape[0]), RUL_hat, label='pred')
plt.plot(np.arange(10-1, RUL_hat.shape[0]), moving_average(RUL_hat, 10), label='pred', alpha=0.5)
plt.xlabel(f'step')
plt.legend()
# plt.yscale('log')
plt.savefig(f'plots/RUL_modelling_inner_model.png')
plt.close()