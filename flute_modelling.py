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


cnc_dataset = CNC_dataset(data_path, machine=machine, columns=data_columns)
init_data = cnc_dataset.get_data()

dataset = TensorDataset(Tensor(init_data.iloc[:-1:1000, :-4].values),
                        Tensor(init_data.iloc[1::1000, :-4].values),
                        Tensor(init_data.iloc[1::1000, -3:].values))
training_loader = DataLoader(dataset, batch_size=model_params['bs'], shuffle=True, drop_last=True)


print('Initializing a model')
model = DeepKoopman(model_params)
optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay=1e-7)


print('Training the model')
for epoch in range(1, model_params['epochs']+1):
    train_loss = train_model(model, optimizer, training_loader)
    print(f'Epoch {epoch}. Train loss {train_loss}')

    torch.save(model.state_dict(), f'models/model_epoch_{epoch}.pth')


print('Evaluation')
# cnc_dataset = CNC_dataset(eval_data_path, machine=eval_machine, columns=data_columns)
# init_data = cnc_dataset.get_data()
eval_data = init_data.iloc[:-1:10000, :]
observables = model.encoder(Tensor(eval_data.iloc[:, :-4].values)).detach()

# plot the evolution of observables
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

w = 100

plt.figure(figsize=(10, 5))
for obs_i in range(model_params['obsdim']):
    plt.plot(eval_data.index[w-1:], moving_average(observables[:, obs_i], w), label=f'obs{obs_i}')
for hp in ['flute_1', 'flute_2', 'flute_3']:
    plt.plot(eval_data.index[w-1:], eval_data.iloc[w-1:,:][hp], label=hp)
plt.legend()
plt.xlabel('step')
plt.savefig(f'plots/observables.png')
plt.close()

# correlation plots of observables and health parameters
for hp in ['flute_1', 'flute_2', 'flute_3']:
    for obs_i in range(model_params['obsdim']):
        plt.figure(figsize=(10, 10))
        plt.scatter(moving_average(observables[:, obs_i], w), eval_data.iloc[w-1:,:][hp])
        plt.xlabel(f'obs dim {obs_i}')
        plt.ylabel(hp)
        plt.savefig(f'plots/scatter_{hp}_obs{obs_i}.png')
        plt.close()

print('Modelling in the observables space')
lr_f1 = Lasso(alpha=100).fit(observables, eval_data['flute_1'])
lr_f2 = Lasso(alpha=100).fit(observables, eval_data['flute_2'])
lr_f3 = Lasso(alpha=100).fit(observables, eval_data['flute_3'])
f1_hat = lr_f1.predict(observables)
f2_hat = lr_f2.predict(observables)
f3_hat = lr_f3.predict(observables)

for hp, prediction in zip(['flute_1', 'flute_2', 'flute_3'], [f1_hat, f2_hat, f3_hat]):
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(eval_data[hp].shape[0]), eval_data[hp], label='ref')
    plt.plot(np.arange(10-1, prediction.shape[0]), moving_average(prediction, 10), label='pred')
    plt.xlabel(f'step')
    plt.legend()
    plt.savefig(f'plots/{hp}_modelled.png')
    plt.close()
