import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

print('Reading the data')
path = 'data/cnc_milling_machine/c1'
cols = ['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz', 'AE-RMS']
file_names = os.listdir(path + '/c1')

wear = pd.read_csv(f'{path}/c1_wear.csv')
datasets = []

for i, filename in enumerate(file_names):
    print(i)
    df = pd.read_csv(f'{path}/c1/{filename}', header=None)
    df.columns = cols
    df['flute_1'] = np.nan
    df['flute_2'] = np.nan
    df['flute_3'] = np.nan
    df.iloc[-1, -3] = wear.loc[i, 'flute_1']
    df.iloc[-1, -2] = wear.loc[i, 'flute_2']
    df.iloc[-1, -1] = wear.loc[i, 'flute_3']
    datasets.append(df)

data = pd.concat(datasets, axis=0)
data = data.reset_index(drop=True)
interp_data = data.dropna()
interp_data = interp_data.reset_index()
flute1_interp = interp1d(interp_data['index'], interp_data['flute_1'], kind='linear', bounds_error=False)
flute2_interp = interp1d(interp_data['index'], interp_data['flute_2'], kind='linear', bounds_error=False)
flute3_interp = interp1d(interp_data['index'], interp_data['flute_3'], kind='linear', bounds_error=False)
data['flute_1'] = flute1_interp(data.index.values)
data['flute_2'] = flute2_interp(data.index.values)
data['flute_3'] = flute3_interp(data.index.values)

data.to_feather('data/cnc_milling_machine/c1/c1.feather')



train_data.columns = cols
train_data = train_data.reset_index(drop=True)
test_data.columns = cols


cols = ['Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Vz', 'AE-RMS']
data = pd.read_csv('data/cnc_milling_machine/c1/c1/c_1_001.csv', header=None)
data.columns = cols
signal = data['Vx'].values
signal = (signal - signal.mean()) / signal.std()
signal = signal[np.newaxis,:,np.newaxis,np.newaxis]

data = pd.read_csv('data/cnc_milling_machine/c1/c1/c_1_002.csv', header=None)
data.columns = cols
testsignal = data['Vx'].values
testsignal = (testsignal - testsignal.mean()) / testsignal.std()
testsignal = testsignal[np.newaxis,:,np.newaxis,np.newaxis]


print('Model initialization')
kernelInit = np.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
                           -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965])
kernTrainable = True
level = 4  # np.floor(np.log2(signal.shape[1])).astype(int)
lossCoeff='l1'
mode = 'PerLayer'
initHT = 0.3
trainHT = True
lossFactor = 1.0

def coeffLoss(yTrue,yPred):
    return lossFactor*tf.reduce_mean(yPred,keepdims=True)
# Set residual loss:
def recLoss(yTrue,yPred):
    return tf.math.abs(yTrue-yPred)

keras.backend.clear_session()
model1,model2 = createDeSpaWN(inputSize=None, kernelInit=kernelInit, kernTrainable=kernTrainable, level=level, lossCoeff=lossCoeff, kernelsConstraint=mode, initHT=initHT, trainHT=trainHT)
opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
model1.compile(optimizer=opt, loss=[recLoss, coeffLoss])

print('Training')
epochs = 100
verbose = 2

H = model1.fit(signal,[signal,np.empty((signal.shape[0]))], epochs=epochs, verbose=verbose)

print('Evaluation')
out  = model1.predict(testsignal)
outC = model2.predict(testsignal)

print('test signal shape', testsignal.shape)
print('outC[1]', outC[1].shape)
print('outC[2]', len(outC[2]))
print('outC[2][0]', outC[2][1].shape)
fig = plt.figure(figsize=(10, 5))
plt.plot(np.arange(100), testsignal[0,:100,0,0], label='original signal')
plt.plot(np.arange(100), out[0][0,:100,0,0], label='preocessed signal')
plt.legend()
plt.savefig('plots/prediction.png', bbox_inches='tight')