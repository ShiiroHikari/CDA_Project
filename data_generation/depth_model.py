#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:08:04 2024

@author: Basile Dupont
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from data_generation import matrix



# Model
class DepthModel(nn.Module):
    def __init__(self, signal_len, num_stations, include_distance=True):
        super(DepthModel, self).__init__()
        
        self.include_distance = include_distance
        
        # Convolutional layers for seismic signals
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Flatten the output
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        conv_out_features = 64 * num_stations * (signal_len // 8) 
        if include_distance:
            conv_out_features += num_stations

        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(conv_out_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)  # Single output for depth
        
    def forward(self, x_signal, x_distance=None):
        # Signal processing through convolutional layers
        x = F.relu(self.bn1(self.conv1(x_signal)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten the convolutional output
        x = self.flatten(x)
        
        # Optionally incorporate distance data
        if self.include_distance and x_distance is not None:
            x = torch.cat((x, x_distance), dim=1)  # Concatenate along feature dimension
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc_out(x)  # Final regression output (no activation)
        
        return x



# Run the model (train, validation and test)
def train_DepthModel(model_name, batch_size=32, num_stations=50, rand_inactive=0, epochs=50, include_distance=True):
    # Prepare train, validation, and test datasets
    X_train, y_train, D_train, signal_shape = matrix.dataset_generation(num_entries=batch_size, num_stations=num_stations, rand_inactive=rand_inactive)
    print("Successfully generated train dataset.")
    
    X_val, y_val, D_val, _ = matrix.dataset_generation(num_entries=batch_size, num_stations=num_stations, rand_inactive=rand_inactive)
    print("Successfully generated validation dataset.")
    
    X_test, y_test, D_test, _ = matrix.dataset_generation(num_entries=batch_size, num_stations=num_stations, rand_inactive=rand_inactive)
    print("Successfully generated test dataset.")

    if include_distance:
        train_dataset = TensorDataset(X_train, y_train, D_train)
        val_dataset = TensorDataset(X_val, y_val, D_val)
        test_dataset = TensorDataset(X_test, y_test, D_test)
    else:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warn user if the device is CPU
    if device.type == "cpu":
        print("CUDA is not available.")
        return None, None, None, None
    
    model = DepthModel(signal_len=signal_shape, num_stations=num_stations, include_distance=include_distance).to(device)
    print(f"Successfully initialized model using {device}.")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print("Successfully set loss criterion and optimizer.")
    
    # Training
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training Epochs", mininterval=1):
        model.train()  # Set model to training mode
        running_loss = 0.0
    
        for batch in train_loader:
            if include_distance:
                X_signal, y_true, x_distance = batch
                X_signal, y_true, x_distance = X_signal.to(device), y_true.to(device), x_distance.to(device)
            else:
                X_signal, y_true = batch
                X_signal, y_true = X_signal.to(device), y_true.to(device)
    
            # Zero gradients
            optimizer.zero_grad()
    
            # Forward pass
            y_pred = model(X_signal, x_distance if include_distance else None)
    
            # Compute loss
            loss = criterion(y_pred, y_true)
    
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
    
            # Accumulate loss for reporting
            running_loss += loss.item()

        # Calculate mean loss for this epoch
        mean_loss = running_loss / len(train_loader)
        train_losses.append(mean_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if include_distance:
                    X_signal, y_true, x_distance = batch
                    X_signal, y_true, x_distance = X_signal.to(device), y_true.to(device), x_distance.to(device)
                else:
                    X_signal, y_true = batch
                    X_signal, y_true = X_signal.to(device), y_true.to(device)
    
                # Forward pass
                y_pred = model(X_signal, x_distance if include_distance else None)
    
                # Compute validation loss
                loss = criterion(y_pred, y_true)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    print("Successfully trained model.")

    # Test evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            if include_distance:
                X_signal, y_true, x_distance = batch
                X_signal, y_true, x_distance = X_signal.to(device), y_true.to(device), x_distance.to(device)
            else:
                X_signal, y_true = batch
                X_signal, y_true = X_signal.to(device), y_true.to(device)
    
            # Forward pass
            y_pred = model(X_signal, x_distance if include_distance else None)
    
            # Compute test loss
            loss = criterion(y_pred, y_true)
            test_loss += loss.item()

    test_loss /= len(test_loader)

    # Save the model
    model_path_name = device.type + "_DepthModel_" + model_name
    model_path = model_path_name + ".pth"
    torch.save(model.state_dict(), 'models/' + model_path)
    print(f'Saved model as "{model_path}".')

    return model_path_name, train_losses, val_losses, test_loss




def run_DepthModel(model_path="cuda_DepthModel.pth", device_name="cuda", num_stations=50, rand_inactive=0, include_distance=True, depth_list=None, plot=False, save_plot=False):
    '''
    Data should have the same parameters (num_stations, include_distance) as the model used.
    
    Parameters:
    - num_stations : number of stations per event
    - include_distance : use stations to epicenter distance to train the model
    - depth : list of depth (m) to generate the data (should have num_entries length)
    '''
    # Get a single matrix
    X_cpu, y, D, signal_shape = matrix.dataset_generation(num_entries=1, num_stations=num_stations, depth_list=depth_list, rand_inactive=rand_inactive)

    # Initialize the model (ensure parameters match the training setup)
    model = DepthModel(signal_len=signal_shape, num_stations=num_stations, include_distance=include_distance)
    
    # Load the trained model weights
    model.load_state_dict(torch.load('models/' + model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode

    # Use the same device as during training
    device = torch.device(device_name)
    model = model.to(device)
    
    X = X_cpu.to(device)
    if include_distance:
        D = D.to(device)
    
    # Get the predicted depth
    with torch.no_grad():  # Disable gradient computation for inference
        if include_distance:
            predicted_depth = model(X, D)
        else:
            predicted_depth = model(X)

    if plot:
        # Plot envelopes
        image = X_cpu.squeeze(0).squeeze(0)  # Turn signals into 2D for mapping
        
        plt.figure(figsize=(10,7))
        plt.imshow(image, aspect='auto', cmap='viridis', origin='upper')
        
        # Adjust x-axis to represent time in seconds
        num_columns = len(image[0])  # Number of columns in the matrix
        plt.xticks(
            ticks=np.arange(0, num_columns, step=200),  # 200 step for 10s spacing
            labels=np.arange(0, num_columns / 20, step=200 / 20)  # Convert to seconds (1/20 of a second since 20 Hz sampling)
        )
    
        plt.xlabel('Time (s)')
        plt.ylabel('Signals (organized by distance)')
        plt.title(f'Real Depth : {y.item()/1e3:.2f} km \nPredicted Depth : {predicted_depth.item()/1e3:.2f} km')
        plt.tight_layout()
        plt.suptitle('Main Energetic envelope of the Z-normalized signals aligned with P-arrival', fontsize=14, fontweight='bold', y=1.02)  # Add suptitle with y offset

        if save_plot:
            plt.savefig(f'figures/{model_path}_comparison.png', bbox_inches='tight')
        
        plt.show()

    # Get missvalue
    delta_depth = predicted_depth.item() - y.item()

    return delta_depth

    
    


