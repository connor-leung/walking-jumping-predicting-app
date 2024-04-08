import matplotlib.pyplot as plt
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

with h5py.File('finalDataset.h5', 'r') as f:
    trainData = f['dataset/Train/trainData'][:]

trainData = pd.DataFrame(trainData)

# jumping = 1, walking = 0
# jumping if position > 6
labels = trainData[0].apply(lambda x: 1 if x > 6 else 0 )

# splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    trainData, labels, test_size=0.1, random_state=42, shuffle=True)

# setting parameters for the logistic regression model
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

# training the model
clf.fit(X_train, Y_train)

# testing the model on the test data from the original dataset
Y_Pred = clf.predict(X_test)
Y_clf_prob = clf.predict_proba(X_test)

# printing out the predicted vs actual values
print(f"{Y_Pred=}")
print(f"{Y_test.to_numpy()=}")

# calculating the accuracy of the model
model_accuracy = accuracy_score(Y_test, Y_Pred)
print(f"{model_accuracy=}")
model_recall = recall_score(Y_test, Y_Pred)

# creating a confusion matrix
cm = confusion_matrix(Y_test, Y_Pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# calculating the F1 score
f1_score = 2 * (model_accuracy * model_recall) / \
    (model_accuracy + model_recall)
print(f"{f1_score=}")

# creating the ROC curve
fpr, tpr, thresholds = roc_curve(
    Y_test, Y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()
roc_auc = roc_auc_score(Y_test, Y_clf_prob[:, 1])
print(f"{roc_auc=}")

