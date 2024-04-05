# Imports
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def windowSegmenting(df):
    """Transforms DataFrame and segments data into 5 second windows"""
    # Select the z acceleration data
    df = df.iloc[:, 3]
    # Drop the label row and transform DataFrame into a list
    values = df.iloc[1:].tolist()

    # List will be returned with the segmented data
    segmentList = []

    # Loops through in intervals equivalent to 5 seconds, segmenting the
    # data and appending to return list
    for i in range(0,len(values), 500):
        segmentList.append(values[i:i+500])

    return segmentList

def shuffle(list):
    """Function utilizes np.random.shuffle to shuffle the list of data segments"""
    # Shuffle
    np.random.shuffle(list)
    return list

# Read in raw data file with pandas function
elizabethDf = pd.read_csv(r"ConnorData.csv")
connorDf = pd.read_csv(r"ElizabethData.csv")

# Segment Elizabeth's Data into the appropriate amount of windows
elizabethSegment = windowSegmenting(elizabethDf)
# Shuffle the segmented windows
elizabethSegment = shuffle(elizabethSegment)

# Segment Connor's Data into the appropriate amount of windows
connorSegment = windowSegmenting(connorDf)
# Shuffle the segmented windows
connorSegment = shuffle(connorSegment)

# Transform both segment lists into a DataFrames
elizabethSegment = [pd.DataFrame(value) for value in elizabethSegment]
connorSegment = [pd.DataFrame(value) for value in connorSegment]
# Combine DataFrames
totalSegment = elizabethSegment + connorSegment
# Put into on data frame with concat
dataframe = pd.concat(totalSegment)
# Shuffle combined DataFrames
shuffle(totalSegment)

# Split the DataFrame into 10% for test and 90% for train
dfTrain, dfTest = train_test_split(dataframe, test_size=0.1, shuffle=False, random_state=42)

# open finalDataset HDF5 file
with h5py.File('finalDataset.h5', 'w') as h5:

    # Create the dataset group file inside the finalDataset file
    datasetGroup = h5.create_group('dataset')

    # Create the train and test files inside the dataset file
    trainGroup = datasetGroup.create_group('Train')
    testGroup = datasetGroup.create_group('Test')

    # Add the test and train data to their associated groups
    testGroup.create_dataset('testData', data = dfTest.to_numpy())
    trainGroup.create_dataset('trainData', data=dfTrain.to_numpy())

    # Create the Group member's data files in the finalDataset file
    elizabethGroup = h5.create_group("Elizabeth's Group")
    connorGroup = h5.create_group("Connor's Group")

    # Add the group member's data into their associated groups
    elizabethGroup.create_dataset("Elizabeth's Data", data = elizabethDf.to_numpy())
    connorGroup.create_dataset("Connor's Data", data = connorDf.to_numpy())
