import numpy as np


machine = 4
data_path = f'../DATASETS/cnc_milling_machine/c{machine}'
eval_machine = 1
eval_data_path = f'../DATASETS/cnc_milling_machine/c{eval_machine}'
data_columns = ['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz', 'AE-RMS']
wear_threshold = 150

model_params_exp1 = {
    'indim': 18,
    'hidden_dim': 100,
    'obsdim': 10,
    'outdim': 18,
    'lr': 0.001,
    'epochs': 10,
    'bs': 1028,
    'n_targets': 1,
}

model_params_exp2 = {
    'indim': 18,
    'hidden_dim': 100,
    'obsdim': 10,
    'outdim': 18,
    'lr': 0.001,
    'epochs': 10,
    'bs': 1028,
    'n_targets': 1,
}

model_params_exp3 = {
    'indim': 18,
    'hidden_dim': 100,
    'obsdim': 10,
    'outdim': 18,
    'lr': 0.001,
    'epochs': 10,
    'bs': 1028,
    'n_targets': 1,
}



control_model_params = {
    'indim': 9,
    'hidden_dim': 50,
    'obsdim': 10,
    'outdim': 9,
    'controldim': 9,
    'lr': 0.001,
    'epochs': 10,
    'bs': 1028,
    'n_targets': 1,  # 3 in case of wear, 1 in case of RUL modelling
}

model_params3 = {
    'indim': 18,
    'hidden_dim': 50,
    'obsdim': 10,
    'outdim': 18,
    'lr': 0.001,
    'epochs': 6,
    'bs': 1028,
    'n_targets': 1,  # 3 in case of wear, 1 in case of RUL modelling
}

despawn_params = {'n_train': 10000,
                  'lossFactor': 1.0,
                  'kernelInit': np.array([-0.010597401785069032,
                                          0.0328830116668852,
                                          0.030841381835560764,
                                          -0.18703481171909309,
                                          -0.027983769416859854,
                                          0.6308807679298589,
                                          0.7148465705529157,
                                          0.2303778133088965]),
                  'kernTrainable': True,
                  'level': 7,
                  'lossCoeff': 'l1',
                  'mode': 'PerLayer',
                  'initHT': 0.3,
                  'trainHT': True,
                  'epochs': 1000,
                  'verbose': 0,
                  }