import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import h5py
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import FeatureExtractionandNormailzation as FE
# Assuming the necessary libraries for model loading and prediction are imported
# For demonstration, actual model loading and prediction logic are simplified

# Placeholder for loading your model

global path

def load_pretrained_model():
    # Load and return your pre-trained model here
    # For demonstration, this will be a placeholder

    with h5py.File('finalDataset.h5', 'r') as f:
        trainData = f['dataset/Train/trainData'][:]

    trainData = pd.DataFrame(trainData)


    # jumping = 1, walking = 0 
    labels = trainData[0].apply(lambda x: 1 if x > 6 else 0 )


    X_train, X_test, Y_train, Y_test = train_test_split(
        trainData, labels, test_size=0.1, random_state=42, shuffle=True)

    l_reg = LogisticRegression(max_iter=10001)
    clf = make_pipeline(StandardScaler(), l_reg)

    # Training.
    clf.fit(X_train, Y_train)
    return clf


# Placeholder model variable
model = load_pretrained_model()

def process_and_predict(input_path):
    global path
    path = input_path
    # Simulated processing and prediction function
    df = pd.read_csv(input_path)
    df = df.iloc[:, 3]
    df = df.iloc[1:]
    df = df.to_frame()
    createFeatureLabels(df)
    # For simplicity, assuming a single feature 'y_acceleration' directly predicts activity
    # Replace with actual data processing and model prediction
    predictions = model.predict(df)
    return df.index, predictions, df



def plot_classifications(time_frames, predictions, graph_frame):
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(10, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.scatter(time_frames, predictions, c=predictions, cmap='viridis', alpha=0.5, s= 0.1)
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Classification')
    ax.set_title('Activity Classification over Time')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Walking', 'Jumping'])

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def plot_empty(graph_frame):
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(10, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Time Frame')
    ax.set_ylabel('Classification')
    ax.set_title('Activity Classification over Time')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Walking', 'Jumping'])

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def open_csv(graph_frame):
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if filepath:
        time_frames, predictions, dataFrame = process_and_predict(filepath)
        plot_classifications(time_frames, predictions, graph_frame)

def createFeatureLabels(dataset):
    features = pd.DataFrame(
        columns=['max', 'min', 'mean', 'median', 'range', 'variance', 'std', 'z-score', 'kurtosis', 'skewness'])
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

    mean_label.config(text='Mean: {:.2f}'.format(features['mean'].iloc[0]))
    std_label.config(text='STD: {:.2f}'.format(features['std'].iloc[0]))
    max_label.config(text='Max: {:.2f}'.format(features['max'].iloc[0]))
    min_label.config(text='Min: {:.2f}'.format(features['min'].iloc[0]))
    median_label.config(text='Median: {:.2f}'.format(features['median'].iloc[0]))
    range_label.config(text='Range: {:.2f}'.format(features['range'].iloc[0]))
    variance_label.config(text='Variance: {:.2f}'.format(features['variance'].iloc[0]))
    zscore_label.config(text='Z-score: {:.2f}'.format(features['z-score'].iloc[0]))
    kurtosis_label.config(text='Kurtosis: {:.2f}'.format(features['kurtosis'].iloc[0]))
    skewness_label.config(text='Skewness: {:.2f}'.format(features['skewness'].iloc[0]))

def download_file():
    filepath = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV files", "*.csv")])

    if not filepath:
        return

    index, predictions, zAccel = process_and_predict(path)
    # Combine index and predictions into a DataFrame
    df = pd.DataFrame({
        'z-acceleration': zAccel.iloc[:, 0],
        'Prediction': predictions
    })

    # Save the DataFrame to CSV
    df.to_csv(filepath, index=False)

def create_gui():
    root = tk.Tk()
    root.title("Activity Classifier")
    root.config(bg= '#FFFFFF')

    # Create frames for the navigation bar and the graph display
    button_frame = tk.Frame(root, width=200, background= '#FFFFFF')
    button_frame.grid(row =0, column = 0, ipadx=5, ipady=5)

    feature_frame = tk.Frame(root, width=200, background='#FFFFFF')
    feature_frame.grid(row=0, column=2, ipadx=5, ipady=5)

    graph_frame = tk.Frame(root, width=600, height=400)
    graph_frame.grid(row =0, column =1, ipadx=5, ipady=5)

    plot_empty(graph_frame)
    # Add an "Open CSV" button to the navigation frame
    open_button = tk.Button(button_frame, text="Open CSV", command=lambda: open_csv(graph_frame),width = 15, height=2, background= '#FFFFFF')
    open_button.grid(row=1, column =0, padx = 5,pady =2,sticky = 'EW')

    close_button = tk.Button(button_frame, text = 'Close', command = lambda: root.destroy(), width = 15, height=2, background= '#FFFFFF')
    close_button.grid(row = 2, column =0, padx = 5, pady =2, sticky = 'EW')

    upload_button = tk.Button(button_frame, text='Download', command = lambda: download_file(), width = 15, height=2, background= '#FFFFFF')
    upload_button.grid(row=3, column=0, padx=5, pady=2, sticky='EW')

    feature_label = tk.Label(feature_frame, text='Features: ', width=20, background= '#FFFFFF', font=(15))
    feature_label.grid(row =0, column =0, sticky='N', pady=10)

    global mean_label
    mean_label = tk.Label(feature_frame, text='Mean: ', width=20, background='#FFFFFF', font=(15))
    mean_label.grid(row=1, column=0, sticky='N', pady=10)

    global std_label
    std_label = tk.Label(feature_frame, text='STD:', width=20, background='#FFFFFF', font=(15))
    std_label.grid(row=2, column=0, sticky='N', pady=10)

    global max_label
    max_label = tk.Label(feature_frame, text='Max:', width=20, background='#FFFFFF', font=(15))
    max_label.grid(row=3, column=0, sticky='N', pady=10)

    global min_label
    min_label = tk.Label(feature_frame, text='Min:', width=20, background='#FFFFFF', font=(15))
    min_label.grid(row=4, column=0, sticky='N', pady=10)

    global median_label
    median_label = tk.Label(feature_frame, text='Median:', width=20, background='#FFFFFF', font=(15))
    median_label.grid(row=5, column=0, sticky='N', pady=10)

    global range_label
    range_label = tk.Label(feature_frame, text='Range:', width=20, background='#FFFFFF', font=(15))
    range_label.grid(row=6, column=0, sticky='N', pady=10)

    global variance_label
    variance_label = tk.Label(feature_frame, text='Variance:', width=20, background='#FFFFFF', font=(15))
    variance_label.grid(row=7, column=0, sticky='N', pady=10)

    global zscore_label
    zscore_label = tk.Label(feature_frame, text='Z score:', width=20, background='#FFFFFF', font=(15))
    zscore_label.grid(row=8, column=0, sticky='N', pady=10)

    global kurtosis_label
    kurtosis_label = tk.Label(feature_frame, text='Kurtosis:', width=20, background='#FFFFFF', font=(15))
    kurtosis_label.grid(row=8, column=0, sticky='N', pady=10)

    global skewness_label
    skewness_label = tk.Label(feature_frame, text='Skewness:', width=20, background='#FFFFFF', font=(15))
    skewness_label.grid(row=9, column=0, sticky='N', pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
