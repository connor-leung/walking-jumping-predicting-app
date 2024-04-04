import DataStoring as ds
from sklearn.preprocessing import StandardScaler

windowSize = 5
filteredDataset = ds.dataframe.rolling(window=windowSize).mean()

scaler = StandardScaler()

dfNormalized = scaler.fit_transform(ds.dataframe)
