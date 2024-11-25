#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 02:48:27 2024

@author: Basile Dupont
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNNModel(nn.Module):
    def __init__(self, X_dim):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Conv2D with 16 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halving the dimensions
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Conv2D with 32 filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Another halving
        )
        
        # Calculate the feature size after convolutions
        feature_size = (50 // 4) * (X_dim // 4) * 32  # Division by 4 (2 successive max-pooling)

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_size, 128),  # Fully connected layer with 128 neurons
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for regression (depth)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x

def initiate_model():
    # Get datasets
    X, y = generate_dataset(num_entries=1000, num_stations=50)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Initiate dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    X_dim = X.shape[3]  # Get second dimension size
    model = CNNModel(X_dim).to('cuda')  # Use GPU for model

    criterion = nn.MSELoss()  # Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return X_tensor, y_tensor, dataset, dataloader, model, criterion, optimizer


def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()  # Put the model in training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')  # Move to GPU

            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backpropagation and weight updates
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")


def test_model(model):
    model.eval()  # Evaluation mode
    with torch.no_grad():
        X_test, y_test = generate_dataset(num_entries=100, num_stations=50)  # Generate test set
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to('cuda')
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to('cuda')
        
        predictions = model(X_test_tensor).squeeze()
        test_loss = criterion(predictions, y_test_tensor)
        print(f"Test Loss: {test_loss.item():.4f}")
