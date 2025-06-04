import streamlit as st
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.graph_objects as go
import pandas as pd

# Define the CNN model (same as provided)
class CNN(nn.Module):
    def __init__(self, n_ch, n_times, n_classes, F1=8, D=2, F2=16, kern_len=64, dropout_rate=0.25):
        super().__init__()
        self.tempConv = nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthConv = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)
        self.sepConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        t_out = n_times // 4 // 8
        self.classify = nn.Linear(F2 * t_out, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.tempConv(x)
        x = self.bn1(x)
        x = self.depthConv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sepConv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.flatten(start_dim=1)
        x = self.classify(x)
        return x

# Function to load the model
def load_model(model_path, n_ch, n_times, n_classes):
    model = CNN(n_ch=n_ch, n_times=n_times, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to preprocess EEG data (same as training)
def preprocess_eeg(data):
    data = torch.from_numpy(data).float().unsqueeze(1)
    data = data - data.mean(dim=-1, keepdim=True)
    data = data / (data.std(dim=-1, keepdim=True) + 1e-8)
    return data

# Streamlit app
st.title("CNN Model Evaluation")
st.write("Evaluate the trained CNN model on the test dataset.")

# File uploaders for test data
cleaned_eeg_file = st.file_uploader("Upload cleaned EEG data (.npy)...", type=["npy"])
labels_file = st.file_uploader("Upload labels (.npy)...", type=["npy"])

# Model parameters
n_ch = st.number_input("Number of EEG channels", min_value=1, value=64, step=1)
n_times = st.number_input("Number of time points", min_value=1, value=1000, step=1)
n_classes = st.number_input("Number of classes", min_value=2, value=10, step=1)

if cleaned_eeg_file is not None and labels_file is not None:
    # Load data
    try:
        X = np.load(cleaned_eeg_file)
        y = np.load(labels_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Validate data shapes
    if len(X.shape) != 3 or X.shape[1:] != (n_ch, n_times):
        st.error(f"Invalid EEG data shape. Expected [n_trials, {n_ch}, {n_times}], got {X.shape}")
        st.stop()
    if len(y.shape) != 1 or len(y) != X.shape[0]:
        st.error(f"Invalid labels shape. Expected [{X.shape[0]}], got {y.shape}")
        st.stop()

    # Split data into train and test sets (same as training script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_test = preprocess_eeg(X_test)
    y_test = torch.from_numpy(y_test).long()

    # Create DataLoader for test set
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    # Load the model
    model_path = "model_cnn.pth"
    try:
        model = load_model(model_path, n_ch, n_times, n_classes)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Evaluate the model
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # Display metrics as a bar chart
    st.subheader("Evaluation Metrics")
    metrics_data = {
        "Metric": ["Test Loss", "Accuracy"],
        "Value": [test_loss, accuracy * 100]  # Accuracy in percentage
    }
    fig_metrics = go.Figure(data=[
        go.Bar(x=metrics_data["Metric"], y=metrics_data["Value"], 
               text=[f"{v:.4f}" if i == 0 else f"{v:.2f}%" for i, v in enumerate(metrics_data["Value"])],
               textposition="auto")
    ])
    fig_metrics.update_layout(
        title="Test Loss and Accuracy",
        yaxis_title="Value",
        yaxis=dict(range=[0, max(metrics_data["Value"]) * 1.2])  # Adjust y-axis range
    )
    st.plotly_chart(fig_metrics)

    # Compute and display confusion matrix as a Plotly heatmap
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(all_labels, all_preds, labels=range(n_classes))
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=range(n_classes),
        y=range(n_classes),
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        hoverinfo="text+z"
    ))
    fig_cm.update_layout(
        title="Confusion Matrix: CNN",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickmode="array", tickvals=range(n_classes)),
        yaxis=dict(tickmode="array", tickvals=range(n_classes))
    )
    st.plotly_chart(fig_cm)

