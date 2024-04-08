# Imports
import DataStoring as ds
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Window size
windowSize = 5
plt.plot(ds.dataframe.index, ds.dataframe, color = "pink")
plt.xlabel("Dataset Index")
plt.ylabel("Linear Acceleration z (m/s^2)")
plt.title("Linear Acceleration Z Data vs Dataset Index")
plt.grid(True)
plt.show()

# Applying the rolling mean filter
filteredDataset = ds.dataframe.rolling(window=windowSize).mean()
filteredDataset = filteredDataset.dropna()

plt.plot(filteredDataset.index, filteredDataset, color = "pink")
plt.xlabel("Dataset Index")
plt.ylabel("Linear Acceleration z (m/s^2)")
plt.title("Filtered Linear Acceleration Z Data vs Dataset Index")
plt.grid(True)
plt.show()

# Initiate the scaler function
scaler = StandardScaler()
# Normalize the data
dfNormalized = scaler.fit_transform(ds.dataframe)
index = np.arange(len(dfNormalized))

plt.plot(index, dfNormalized, color = "pink")
plt.xlabel("Dataset Index")
plt.ylabel("Linear Acceleration z (m/s^2)")
plt.title("Normalized Linear Acceleration Z Data vs Dataset Index")
plt.grid(True)
plt.show()