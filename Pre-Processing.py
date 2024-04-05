# Imports
import DataStoring as ds
from sklearn.preprocessing import StandardScaler

# Window size
windowSize = 5

# Applying the rolling mean filter
filteredDataset = ds.dataframe.rolling(window=windowSize).mean()
filteredDataset = filteredDataset.dropna()

# Initiate the scaler function
scaler = StandardScaler()
# Normalize the data
dfNormalized = scaler.fit_transform(ds.dataframe)
