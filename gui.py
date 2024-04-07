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
# Assuming the necessary libraries for model loading and prediction are imported
# For demonstration, actual model loading and prediction logic are simplified

# Placeholder for loading your model
def load_pretrained_model():
    # Load and return your pre-trained model here
    # For demonstration, this will be a placeholder

    with h5py.File('finalDataset.h5', 'r') as f:
        trainData = f['dataset/Train/trainData'][:]

    trainData = pd.DataFrame(trainData)


    # jumping = 1, walking = 0
    # jumping if position > 1.5 or position < -1.5
    trainData[0] = trainData[0].apply(lambda x: 1 if x < -1.5 or x > 1.5 else 0 )

    labels = trainData[0]

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
    # Simulated processing and prediction function
    df = pd.read_csv(input_path)
    # For simplicity, assuming a single feature 'y_acceleration' directly predicts activity
    # Replace with actual data processing and model prediction
    predictions = model.predict(df['y_acceleration'])

    return df.index, predictions

def plot_classifications(time_frames, predictions, graph_frame):
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig = Figure(figsize=(10, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.scatter(time_frames, predictions, c=predictions, cmap='viridis', alpha=0.5)
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
        filepath = filepath.iloc[:, 3]

        time_frames, predictions = process_and_predict(filepath)
        plot_classifications(time_frames, predictions, graph_frame)

def create_gui():
    root = tk.Tk()
    root.title("Activity Classifier")

    # Create frames for the navigation bar and the graph display
    nav_frame = tk.Frame(root, width=200, bg='grey')
    graph_frame = tk.Frame(root, width=600, height=400)

    nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Add an "Open CSV" button to the navigation frame
    open_button = tk.Button(nav_frame, text="Open CSV", command=lambda: open_csv(graph_frame))
    open_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
