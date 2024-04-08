import pandas as pd
import h5py

# loading the dataset
with h5py.File('finalDataset.h5', 'r') as hf:
    data = hf['dataset/Train/trainData'][:]

dataset = pd.DataFrame(data)

# creating the feature labels
features = pd.DataFrame(columns=['max', 'min', 'mean', 'median', 'range', 'variance', 'std', 'z-score', 'kurtosis', 'skewness'])

# calculates mean
features['mean'] = dataset.mean()

# calculates standard deviation
features['std'] = dataset.std()

# calculates maximum value
features['max'] = dataset.max()

# calculates minimum value
features['min'] = dataset.min()

# calculates median
features['median'] = dataset.median()

# calculates range
features['range'] = features['max'] - features['min']

# calculates variance
features['variance'] = dataset.var()

# calculates z-score
features['z-score'] = (dataset - features['mean']) / features['std']

# calculates kurtosis
features['kurtosis'] = dataset.kurt()

# calculates skewness
features['skewness'] = dataset.skew()

print(features)

# normalizing the data
newDataPrime = dataset - features['mean']
newDataBar = newDataPrime / features['std']

# setting dataset to the normalized data
dataset = pd.DataFrame(newDataBar)

# calculating all the features with the normalized data
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
