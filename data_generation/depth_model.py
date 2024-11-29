#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 18:08:04 2024

@author: Basile Dupont
"""

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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Flatten the output
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        conv_out_features = 128 * num_stations * (signal_len // 8) 
        if include_distance:
            conv_out_features += num_stations
            
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
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_out(x)  # Final regression output (no activation)
        
        return x



# Run the model (train and test)
def run_DepthModel(batch_size=32, num_stations=50, epochs=50, include_distance=True):
    # Prepare train and test datasets
    X, y, D, signal_shape = matrix.dataset_generation(num_entries=batch_size, num_stations=num_stations)
    print("Succesfully generated train dataset.")
    
    X_test, y_test, D_test, _ = matrix.dataset_generation(num_entries=batch_size, num_stations=num_stations)
    print("Succesfully generated test dataset.")

    if include_distance:
        dataset = TensorDataset(X, y, D)
        test_dataset = TensorDataset(X_test, y_test, D_test)
    else:
        dataset = TensorDataset(X, y)
        test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Warn user if the device is CPU
    if device.type == "cpu":
        confirm = input("CUDA is not available. The model will run on the CPU, which may be slow. Do you want to continue? (Y/N): ").strip().lower()
        if confirm != "Y":
            print("User declined to proceed on CPU.")
            return None, [X, y, D], [X_test, y_test, D_test]
    
    model = DepthModel(signal_len=signal_shape, num_stations=num_stations, include_distance=include_distance).to(device)
    print(f"Succesfully initialized model using {device}.")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Succesfully set loss criterion and optimizer.")
    
    # Training
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for batch in train_loader:
            if include_distance:
                X_signal, y_true, x_distance = batch
                X_signal, y_true, x_distance = X_signal.to(device), y_true.to(device), x_distance.to(device)
                y_pred = model(X_signal, x_distance)
            else:
                X_signal, y_true = batch
                X_signal, y_true = X_signal.to(device), y_true.to(device)
                y_pred = model(X_signal)
    
            # Zero gradients
            optimizer.zero_grad()
    
            # Forward pass
            y_pred = model(X_signal, x_distance)
    
            # Compute loss
            loss = criterion(y_pred, y_true)
    
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for reporting
            running_loss += loss.item()
    
        print(f"Epoch {epoch+1}/{epochs}, Mean Loss: {running_loss / len(train_loader):.4f}")

    print("Succesfull training.")
    
    # Vspace between training and evaluation
    print("\n \n \n")

    # Evaluation
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation
        for batch in test_loader:
            if include_distance:
                X_signal, y_true, x_distance = batch
                X_signal, y_true, x_distance = X_signal.to(device), y_true.to(device), x_distance.to(device)
                y_pred = model(X_signal, x_distance)
            else:
                X_signal, y_true = batch
                X_signal, y_true = X_signal.to(device), y_true.to(device)
                y_pred = model(X_signal)
    
            # Forward pass
            y_pred = model(X_signal, x_distance)
    
            # Compute loss
            loss = criterion(y_pred, y_true)
            test_loss += loss.item()
    
    print(f"Mean Test Loss: {test_loss / len(test_loader):.4f}")
    print("Succesfull test.")

    # Save the model
    # torch.save(model.state_dict(), "depth_model.pth")

    return model, [X, y, D], [X_test, y_test, D_test]


