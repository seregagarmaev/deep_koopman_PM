import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class CNC_dataset:
    def __init__(self, path, machine=1, columns=['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz', 'AE-RMS'], use_RUL=False, wear_threshold=0):
        self.final_df_path = f'{path}/c{machine}.feather'
        if self.processing_done():
            print(f'Reading preprocessed data from {self.final_df_path}.')
            self.data = pd.read_feather(self.final_df_path)
        else:
            print(f'Processing CNC milling dataset for machine {machine}.')
            self.path = path
            self.machine = machine
            self.columns = columns
            self.wear = pd.read_csv(f'{path}/c{machine}_wear.csv')
            self.filenames = os.listdir(f'{path}/c{machine}')
            self.data = self.load_data()
            self.data = self.interpolate_wear(self.data)
            self.save_data()
            print(f'Saved processed data to {self.final_df_path}.')
        if use_RUL:
            threshold_idx = (self.data[['flute_1', 'flute_2', 'flute_3']] <= wear_threshold).any(axis=1).sum()
            self.data['RUL'] = 0
            self.data.iloc[:threshold_idx, -1] = np.arange(threshold_idx, 0, -1)
            self.data = self.data.drop(['flute_1', 'flute_2', 'flute_3'], axis=1)
            self.data = self.data[self.data['RUL'] > 0]


    def load_data(self):
        datasets = []
        for i, filename in enumerate(self.filenames):
            df = self.read_csv(filename)
            df = self.add_wear(df, i)
            datasets.append(df)
        return pd.concat(datasets, axis=0).reset_index(drop=True)

    def add_wear(self, data, measurement_number):
        data['flute_1'] = np.nan
        data['flute_2'] = np.nan
        data['flute_3'] = np.nan
        data.iloc[-1, -3] = self.wear.loc[measurement_number, 'flute_1']
        data.iloc[-1, -2] = self.wear.loc[measurement_number, 'flute_2']
        data.iloc[-1, -1] = self.wear.loc[measurement_number, 'flute_3']
        return data

    def read_csv(self, filename):
        data = pd.read_csv(f'{self.path}/c{self.machine}/{filename}', header=None)
        data.columns = self.columns
        return data

    def interpolate_wear(self, data):
        interp_data = data.dropna()
        interp_data = interp_data.reset_index()
        flute1_interp = interp1d(interp_data['index'], interp_data['flute_1'], kind='linear', bounds_error=False)
        flute2_interp = interp1d(interp_data['index'], interp_data['flute_2'], kind='linear', bounds_error=False)
        flute3_interp = interp1d(interp_data['index'], interp_data['flute_3'], kind='linear', bounds_error=False)
        data['flute_1'] = flute1_interp(data.index.values)
        data['flute_2'] = flute2_interp(data.index.values)
        data['flute_3'] = flute3_interp(data.index.values)
        return data.dropna().reset_index(drop=True)

    def processing_done(self):
        return os.path.isfile(self.final_df_path)

    def save_data(self):
        self.data.to_feather(self.final_df_path)
        return None

    def get_data(self):
        return self.data