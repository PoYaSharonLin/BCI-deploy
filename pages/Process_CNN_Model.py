import streamlit as st
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import plotly.graph_objects as go
from collections import Counter

# CNN class (unchanged, matches training code)
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

# Function to resize EEG data (unchanged)
def resize_eeg_data(data, expected_times):
    n_trials, n_ch, current_times = data.shape
    if current_times == expected_times:
        return data
    resized_data = np.zeros((n_trials, n_ch, expected_times))
    for trial in range(n_trials):
        for ch in range(n_ch):
            resized_data[trial, ch] = np.interp(
                np.linspace(0, current_times - 1, expected_times),
                np.arange(current_times),
                data[trial, ch]
            )
    return resized_data

# Function to filter rare classes (unchanged)
def filter_rare_classes(X, y, min_samples=2):
    class_counts = Counter(y)
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
    mask = np.isin(y, valid_classes)
    return X[mask], y[mask]

# Updated load_model function
def load_model(model_path, n_ch, n_times, n_classes):
    model = CNN(n_ch=n_ch, n_times=n_times, n_classes=n_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'classify.weight' in state_dict:
        expected_in_features = model.classify.in_features
        actual_in_features = state_dict['classify.weight'].shape[1]
        if actual_in_features != expected_in_features:
            st.error(
                f"Mismatch in linear layer input features: expected {expected_in_features}, "
                f"got {actual_in_features}. Please input the n_times used during training."
            )
            st.stop()
        trained_n_classes = state_dict['classify.weight'].shape[0]
        if trained_n_classes != n_classes:
            state_dict.pop('classify.weight', None)
            state_dict.pop('classify.bias', None)
            st.warning(
                f"Number of classes in model ({trained_n_classes}) differs from data "
                f"({n_classes}). Linear layer weights will be randomly initialized."
            )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Function to preprocess EEG data (unchanged)
def preprocess_eeg(data):
    data = torch.from_numpy(data).float().unsqueeze(1)
    data = data - data.mean(dim=-1, keepdim=True)
    data = data / (data.std(dim=-1, keepdim=True) + 1e-8)
    return data

# Function to display evaluation metrics (unchanged)
def display_evaluation_metrics(avg_test_loss, all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    metrics_data = {
        "Metric": ["Test Loss", "Accuracy", "F1 Macro", "Precision", "Recall"],
        "Value": [avg_test_loss, accuracy * 100, f1 * 100, precision * 100, recall * 100]
    }
    fig_metrics = go.Figure(data=[
        go.Bar(
            x=metrics_data["Metric"],
            y=metrics_data["Value"],
            text=[f"{v:.4f}" if i == 0 else f"{v:.2f}%" for i, v in enumerate(metrics_data["Value"])],
            textposition="auto",
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
    ])
    fig_metrics.update_layout(
        title="Evaluation Metrics (3-Fold Cross-Validation)",
        yaxis_title="Value",
        yaxis=dict(range=[0, max(metrics_data["Value"]) * 1.2]),
        xaxis_title="Metric",
        template="plotly_white"
    )
    st.plotly_chart(fig_metrics)

# Streamlit app
st.title("Process CNN Model Evaluation")
st.write(
    "Evaluate the trained CNN model. The .npy file should be trained based on the CNN model "
    "structure in this repo: https://github.com/bellapd/MNISTMindBigData/blob/main/src/classifier/cnn.py. "
    "Ensure n_times matches the value used during training."
)

# File uploaders for test data
cleaned_eeg_file = st.file_uploader("Upload cleaned EEG data (.npy)...", type=["npy"])
labels_file = st.file_uploader("Upload labels (.npy)...", type=["npy"])

# Model parameters
n_ch = st.number_input("Number of EEG channels", min_value=1, value=14, step=1)
n_times = st.number_input("Number of time points (must match training)", min_value=1, value=256, step=1)
n_classes = st.number_input("Number of classes", min_value=2, value=3, step=1)

if cleaned_eeg_file is not None and labels_file is not None:
    # Load data
    try:
        X = np.load(cleaned_eeg_file)
        y = np.load(labels_file)
        st.write("Class distribution:", dict(Counter(y)))
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Resize EEG data to match n_times
    X = resize_eeg_data(X, n_times)

    # Filter classes with fewer than 2 samples
    X, y = filter_rare_classes(X, y, min_samples=2)
    n_classes = len(np.unique(y))
    st.info(f"Updated number of classes after filtering: {n_classes}")

    # Validate data shapes
    if len(X.shape) != 3 or X.shape[1:] != (n_ch, n_times):
        st.error(f"Invalid EEG data shape. Expected [n_trials, {n_ch}, {n_times}], got {X.shape}")
        st.stop()
    if len(y.shape) != 1 or len(y) != X.shape[0]:
        st.error(f"Invalid labels shape. Expected [{X.shape[0]}], got {y.shape}")
        st.stop()

    # Load the model
    model_path = "model_cnn_process.pth"
    try:
        model = load_model(model_path, n_ch, n_times, n_classes)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    test_losses = []
    all_preds = []
    all_labels = []

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    try:
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_test = preprocess_eeg(X_test)
            y_test = torch.from_numpy(y_test).long()

            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

            # Evaluate the model
            test_loss = 0.0
            fold_preds = []
            fold_labels = []

            with torch.no_grad():
                for xb, yb in test_loader:
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    test_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    fold_preds.extend(preds.numpy())
                    fold_labels.extend(yb.numpy())
            
            test_loss /= len(test_loader) if len(test_loader) > 0 else 1
            test_losses.append(test_loss)
            all_preds.extend(fold_preds)
            all_labels.extend(fold_labels)
    except ValueError as e:
        st.error(f"Error during cross-validation: {e}")
        st.stop()

    # Compute average test loss
    avg_test_loss = np.mean(test_losses)

    # Display evaluation metrics
    st.subheader("Evaluation Metrics")
    display_evaluation_metrics(avg_test_loss, all_labels, all_preds)

    # Compute and display confusion matrix
    st.subheader("Confusion Matrix (All Folds)")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=list(range(n_classes)),
        y=list(range(n_classes)),
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        hoverinfo="text"
    ))
    fig_cm.update_layout(
        title="Confusion Matrix (3-Fold CV)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickmode="array", tickvals=list(range(n_classes))),
        yaxis=dict(tickmode="array", tickvals=list(range(n_classes)))
    )
    st.plotly_chart(fig_cm)
else:
    st.error("Please upload both EEG data and labels files to proceed.")


    