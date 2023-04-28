import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow.keras as keras
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from despawn.despawn import createDeSpaWN
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def train_model_exp1(model, optimizer, dataloader):
    '''
    Trains the model for a single epoch.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()
    final_supervised_loss = 0
    final_reconstruction_loss = 0

    for x, hp in dataloader:
        # here we calculate supervised loss
        obs = model.encoder(x)
        hp_hat = model.lr(obs)
        supervised_loss = mseLoss(hp, hp_hat) + 0.01 * torch.norm(model.lr.weight, p=1)

        # reconstruction loss
        x_hat = model.decoder(model.encoder(x))
        reconstruction_loss = mseLoss(x, x_hat)

        loss = reconstruction_loss + supervised_loss
        final_supervised_loss += supervised_loss.item()
        final_reconstruction_loss += reconstruction_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.detach()
    return total_loss, (final_supervised_loss, final_reconstruction_loss)

def train_model_exp2(model, optimizer, dataloader):
    '''
    Trains the model for a single epoch.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()
    final_supervised_loss = 0
    final_prediction_loss = 0

    for x0, x1, hp in dataloader:
        # here we calculate supervised loss
        obs = model.encoder(x0)
        hp_hat = model.lr(obs)
        supervised_loss = mseLoss(hp, hp_hat) + 0.01 * torch.norm(model.lr.weight, p=1)

        # prediction loss
        x1_hat = model.decoder(model.encoder(x0))
        prediction_loss = mseLoss(x1, x1_hat)

        loss =  supervised_loss + prediction_loss
        final_supervised_loss += supervised_loss.item()
        final_prediction_loss += prediction_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.detach()
    return total_loss, (final_supervised_loss, final_prediction_loss)

def train_model_exp3(model, optimizer, dataloader):
    '''
    Trains the model for a single epoch.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()
    final_supervised_loss = 0
    final_prediction_loss = 0
    final_reconstruction_loss = 0

    for x0, x1, hp in dataloader:
        # here we calculate supervised loss
        obs = model.encoder(x0)
        hp_hat = model.lr(obs)
        supervised_loss = mseLoss(hp, hp_hat) + 0.01 * torch.norm(model.lr.weight, p=1)

        # prediction loss
        x1_hat = model(x0)
        prediction_loss = mseLoss(x1, x1_hat)

        # reconstruction loss
        x0_hat = model.decoder(model.encoder(x0))
        reconstruction_loss = mseLoss(x0, x0_hat)

        loss =  supervised_loss + prediction_loss
        final_supervised_loss += supervised_loss.item()
        final_prediction_loss += prediction_loss.item()
        final_reconstruction_loss += reconstruction_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.detach()
    return total_loss, (final_supervised_loss, final_prediction_loss, final_reconstruction_loss)





def custom_rul_metric(rul_hat, w=100):
    n = len(rul_hat)
    metric = []
    for i in range(0, n, w):
        rul_hat_slice = rul_hat[i:i+w].reshape(-1, 1)
        x = np.arange(i, i+rul_hat_slice.shape[0], 1).reshape(-1, 1)
        lr = LinearRegression().fit(x, rul_hat_slice)
        prediction = lr.predict(x)
        b = lr.intercept_
        a = (prediction[10] - prediction[0]) / 10
        x0 = -b / a
        error = x0 - n
        metric.append(error)
        plt.figure(figsize=(10, 10))
        plt.plot(x, prediction, label='pred')
        plt.plot(np.arange(len(rul_hat)), rul_hat, label='true', alpha=0.1)
        plt.savefig(f'plots/experiment_1/temp/pred_{i}.png')
        plt.close()

    return metric






































def train_model(model, optimizer, dataloader):
    '''
    Trains model for a single epoch.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()

    for x0, x1, hp1 in dataloader:
        # here we calculate supervised loss
        y0 = model.encoder(x0)
        y1 = model.K(y0)
        hp1_hat = model.lr(y0)
        supervised_loss = mseLoss(hp1, hp1_hat)


        # regularization of the supervised loss
        supervised_loss_l1 = torch.norm(model.lr.weight, p=1)

        # autoencoder loss
        x0_hat = model.decoder(model.encoder(x0))
        autoencoder_loss = mseLoss(x0, x0_hat)

        # one step prediction loss
        x1_hat = model(x0.to(device))
        prediction_loss = mseLoss(x1, x1_hat)

        loss = prediction_loss + autoencoder_loss + supervised_loss + supervised_loss_l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.detach()
    return total_loss

def train_control_model(model, optimizer, dataloader):
    '''
    Trains the control model for a single epoch.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()
    final_pred_loss = 0
    final_autoencoder_loss = 0
    final_supervised_loss = 0

    for x0, x1, u0, hp1 in dataloader:
        # here we calculate supervised loss
        y0 = model.encoder(x0)
        Bu0 = model.B(u0)

        obs = torch.cat((y0, Bu0), dim=1)

        hp1_hat = model.lr(obs)
        supervised_loss = mseLoss(hp1, hp1_hat)

        # regularization of the supervised loss
        supervised_loss_l1 = torch.norm(model.lr.weight, p=1)

        # autoencoder loss
        x0_hat = model.decoder(model.encoder(x0))
        autoencoder_loss = mseLoss(x0, x0_hat)

        # one step prediction loss
        x1_hat = model(x0.to(device), u0.to(device))
        prediction_loss = mseLoss(x1, x1_hat)

        loss = prediction_loss + autoencoder_loss + supervised_loss + 0.01 * supervised_loss_l1
        final_pred_loss += prediction_loss.item()
        final_autoencoder_loss += autoencoder_loss.item()
        final_supervised_loss += supervised_loss.item() + 0.01 * supervised_loss_l1.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.detach()
    return total_loss, (final_pred_loss, final_autoencoder_loss, final_supervised_loss)

def train_model_2(model, optimizer, dataloader):
    '''
    Trains the control model for a single epoch.
    The outputs of state vector and control vector's encoders are summed and passed to the Koopman operator.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()
    final_pred_loss = 0
    final_autoencoder_loss = 0
    final_supervised_loss = 0

    for x0, x1, u0, hp1 in dataloader:
        # here we calculate supervised loss
        y0 = model.encoder(x0)
        Bu0 = model.B(u0)
        obs = y0 + Bu0
        hp1_hat = model.lr(obs)
        supervised_loss = mseLoss(hp1, hp1_hat)

        # regularization of the supervised loss
        supervised_loss_l1 = torch.norm(model.lr.weight, p=1)

        # autoencoder loss
        x0_hat = model.decoder(model.encoder(x0))
        autoencoder_loss = mseLoss(x0, x0_hat)

        # one step prediction loss
        x1_hat = model(x0.to(device), u0.to(device))
        prediction_loss = mseLoss(x1, x1_hat)

        loss = prediction_loss + autoencoder_loss + supervised_loss + 0.01 * supervised_loss_l1
        final_pred_loss += prediction_loss.item()
        final_autoencoder_loss += autoencoder_loss.item()
        final_supervised_loss += supervised_loss.item() + 0.01 * supervised_loss_l1.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.detach()
    return total_loss, (final_pred_loss, final_autoencoder_loss, final_supervised_loss)

def train_model_3(model, optimizer, dataloader):
    '''
    Trains the control model for a single epoch.
    The state and control vectors are encoded and passed to the Koopman operator.
    '''
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    mseLoss = nn.MSELoss()
    final_pred_loss = 0
    final_autoencoder_loss = 0
    final_supervised_loss = 0

    for x0, x1, hp1 in dataloader:
        # here we calculate supervised loss
        y0 = model.encoder(x0)
        hp1_hat = model.lr(y0)
        supervised_loss = mseLoss(hp1, hp1_hat)

        # regularization of the supervised loss
        supervised_loss_l1 = torch.norm(model.lr.weight, p=1)

        # autoencoder loss
        x0_hat = model.decoder(model.encoder(x0))
        autoencoder_loss = mseLoss(x0, x0_hat)

        # one step prediction loss
        x1_hat = model(x0.to(device))
        prediction_loss = mseLoss(x1, x1_hat)

        loss = prediction_loss + autoencoder_loss + supervised_loss + 0.01 * supervised_loss_l1
        final_pred_loss += prediction_loss.item()
        final_autoencoder_loss += autoencoder_loss.item()
        final_supervised_loss += supervised_loss.item() + 0.01 * supervised_loss_l1.item()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.detach()
    return total_loss, (final_pred_loss, final_autoencoder_loss, final_supervised_loss)

class DESPAWN_preprocessor:
    def __init__(self, despawn_params):
        self.despawn_params = despawn_params

    def fit_transform(self, dataframe):
        result_dataframe = pd.DataFrame()
        for column in dataframe.columns:
            signal = dataframe[column].values[np.newaxis, :, np.newaxis, np.newaxis]
            train_signal = signal[:, :self.despawn_params['n_train'], :, :]

            self.model1, self.model2, self.model3 = self.prepare_models()

            self.model1, log = self.fit_model(self.model1, train_signal)

            signal_hat, train_loss  = self.model1.predict(signal)
            _, lp_coeff, hp_coeff = self.model2.predict(signal)
            (signal_shapes, kernelsHT, kernelsGT) = self.model3.predict(signal)

            mean_coeffs, max_coeffs, abs_rec_error = self.prepare_features(lp_coeff, hp_coeff, signal_shapes, signal, signal_hat)

            result_dataframe[f'{column}_avg_coeff'] = mean_coeffs
            result_dataframe[f'{column}_max_coeff'] = max_coeffs
            result_dataframe[f'{column}_abs_rec_error'] = abs_rec_error
        return result_dataframe

    def transform(self, dataframe):
        result_dataframe = pd.DataFrame()
        for column in dataframe.columns:
            signal = dataframe[column].values[np.newaxis, :, np.newaxis, np.newaxis]

            signal_hat, train_loss = self.model1.predict(signal)
            _, lp_coeff, hp_coeff = self.model2.predict(signal)
            (signal_shapes, kernelsHT, kernelsGT) = self.model3.predict(signal)

            mean_coeffs, max_coeffs, abs_rec_error = self.prepare_features(lp_coeff, hp_coeff, signal_shapes, signal,
                                                                           signal_hat)

            result_dataframe[f'{column}_avg_coeff'] = mean_coeffs
            result_dataframe[f'{column}_max_coeff'] = max_coeffs
            result_dataframe[f'{column}_abs_rec_error'] = abs_rec_error
        return result_dataframe

    def prepare_features(self, lp_coeff, hp_coeff, signal_shapes, signal, signal_hat):
        coeffs_matrix = []
        lp_coeffs = self.distribute_coeffs(lp_coeff.flatten(),
                                           self.despawn_params['level'] - 1,
                                           [el[1] for el in signal_shapes])
        coeffs_matrix.append(lp_coeffs)

        for level in range(self.despawn_params['level']):
            hp_coeffs = self.distribute_coeffs(hp_coeff[level].flatten(),
                                               level,
                                               [el[1] for el in signal_shapes])
            coeffs_matrix.append(hp_coeffs)

        del lp_coeff, hp_coeff
        coeffs_matrix = np.abs(np.array(coeffs_matrix))

        return np.mean(coeffs_matrix, axis=0), np.max(coeffs_matrix, axis=0), np.abs(signal.flatten() - signal_hat.flatten())[::2]


    def extend_1d_array(self, a, size):
        matrix = np.array([a, a])
        extended = matrix.flatten('F')
        bounded = extended[:size]
        return bounded

    def distribute_coeffs(self, initial_coeffs, level, sizes):
        coeffs = initial_coeffs
        for l in range(level, 0, -1):
            coeffs = self.extend_1d_array(coeffs, sizes[l])
        return coeffs

    def prepare_losses(self):
        # Set sparsity (dummy) loss:
        def coeffLoss(yTrue, yPred):
            return self.despawn_params['lossFactor'] * tf.reduce_mean(yPred, keepdims=True)

        # Set residual loss:
        def recLoss(yTrue, yPred):
            return tf.math.abs(yTrue - yPred)
        return coeffLoss, recLoss

    def prepare_models(self):
        keras.backend.clear_session()
        coeffLoss, recLoss = self.prepare_losses()
        model1, model2, model3 = createDeSpaWN(inputSize=None,
                                               kernelInit=self.despawn_params['kernelInit'],
                                               kernTrainable=self.despawn_params['kernTrainable'],
                                               level=self.despawn_params['level'],
                                               lossCoeff=self.despawn_params['lossCoeff'],
                                               kernelsConstraint=self.despawn_params['mode'],
                                               initHT=self.despawn_params['initHT'],
                                               trainHT=self.despawn_params['trainHT'])

        opt = keras.optimizers.Nadam(learning_rate=0.001,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-07,
                                     name="Nadam")

        model1.compile(optimizer=opt, loss=[recLoss, coeffLoss])
        return model1, model2, model3

    def fit_model(self, model, signal):
        log = model.fit(signal,
                        [signal, np.empty((signal.shape[0]))],
                        epochs=self.despawn_params['epochs'],
                        verbose=self.despawn_params['verbose'])
        return model, log