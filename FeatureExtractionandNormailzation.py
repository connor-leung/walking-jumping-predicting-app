# extract min 10 diff features
import pandas as pd
import numpy as np
import h5py

with h5py.File('finalDataset.h5', 'r') as hf:
    data = hf['dataset/Train/trainData'][:]

dataset = pd.DataFrame(data)

features = pd.DataFrame(columns=['max', 'min', 'mean', 'median', 'range', 'variance', 'std', 'z-score', 'kurtosis', 'skewness'])
features['mean'] = dataset.mean()
features['std'] = dataset.std()
features['max'] = dataset.max()
features['min'] = dataset.min()
features['median'] = dataset.median()
features['range'] = features['max'] - features['min']
features['variance'] = dataset.var()
features['z-score'] = (dataset - features['mean']) / features['std']
features['kurtosis'] = dataset.kurt()
features['skewness'] = dataset.skew()

print(features)

newDataPrime = dataset - features['mean']
newDataBar = newDataPrime / features['std']

dataset = pd.DataFrame(newDataBar)

features['mean'] = dataset.mean()
features['std'] = dataset.std()
features['max'] = dataset.max()
features['min'] = dataset.min()
features['median'] = dataset.median()
features['range'] = features['max'] - features['min']
features['variance'] = dataset.var()
features['z-score'] = (dataset - features['mean']) / features['std']
features['kurtosis'] = dataset.kurt()
features['skewness'] = dataset.skew()

print(features)