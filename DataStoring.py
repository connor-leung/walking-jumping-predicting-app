
import h5py
import pandas
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

def windowSegmenting(df):
    df = df.iloc[:, 3]
    values = df.iloc[1:].tolist()
    segmentList = []
    for i in range(0,len(values), 500):
        segmentList.append(values[i:i+500])
    return segmentList

def shuffle(list):
    np.random.shuffle(list)
    return list

# Get files
dfFirst = pd.read_csv(r"ConnorData.csv")
dfSecond= pd.read_csv(r"ElizabethData.csv")

firstSegment = windowSegmenting(dfFirst)
shuffle(firstSegment)

secondSegment = windowSegmenting(dfSecond)
shuffle(secondSegment)

firstSegment = [pd.DataFrame(value) for value in firstSegment]
secondSegment = [pd.DataFrame(value) for value in secondSegment]

combinedSegment = firstSegment + secondSegment
shuffle(combinedSegment)

# Put into on data frame w concat
dataframe = pd.concat(combinedSegment)
print(dataframe)

# split into90, 10
dfTrain, dfTest = train_test_split(dataframe, test_size=0.1, shuffle=False, random_state=42)

# Open create group
with h5py.File('finalDataset.h5', 'w') as h5:
    datasetGroup = h5.create_group('dataset')
    trainGroup = datasetGroup.create_group('Train')
    testGroup = datasetGroup.create_group('Test')

    testGroup.create_dataset('testData', data = dfTest.to_numpy())
    trainGroup.create_dataset('trainData', data=dfTrain.to_numpy())

    elizabethGroup = h5.create_group("Elizabeth's Group")
    connorGroup = h5.create_group("Connor's Group")

    elizabethGroup.create_dataset("Elizabeth's Data", data = dfFirst.to_numpy())
    connorGroup.create_dataset("Connor's Data", data = dfSecond.to_numpy())
