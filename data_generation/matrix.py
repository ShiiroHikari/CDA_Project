#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 23:57:18 2024

@author: Basile Dupont
"""


import numpy as np
import data_generation.signal

# Generate one matrix
def generate_matrix(num_stations=50):
    """
    Generates a matrix and depth associated.
    
    Parameters:
    - num_entries : number of cases (different depths) to generate
    - num_stations : number of stations per depth
    
    Returns:
    - signal_matrix : matrix with one line per signal
    - depth : the depth corresponding to this matrix
    """
    results = data_generation.signal.generate_signals(num_stations=num_stations)

    # Get depth (same for all)
    depth = results[0][1][2] # from 1st sample

    # Initialize matrix
    num_samples = len(results[0][0])
    signal_matrix = np.zeros((num_stations, num_samples))
    
    # Build matrix
    for i, (envelope, _, _) in enumerate(results):
        signal_matrix[i, :] = envelope

    return signal_matrix, depth


# Generate multiple matrix for model training
def data_generationset(num_entries=1000, num_stations=50):
    """
    Generates a dataset containing signal matrices and their corresponding depths.
    
    Parameters:
    - num_entries : number of cases (different depths) to generate
    - num_stations : number of stations per depth
    
    Returns:
    - X : numpy array of shape (num_entries, 1, num_stations, X) (signal matrices)
    - y : numpy array of shape (num_entries,) (depths)
    """
    data_matrix = []
    data_depth = [] 
    
    for i in range(num_entries):
        # Generate signal matrix and depth
        signal_matrix, depth = generate_matrix(num_stations=num_stations)
        
        # Save matrix and depth
        data_matrix.append(signal_matrix)
        data_depth.append(depth)

    #
    X_dim = data_matrix[0].shape[1]  # Get number of columns (2nd dimension)
    
    # Convert to numpy arrays
    X = np.array(data_matrix).reshape(num_entries, 1, num_stations, X_dim)  # Add channel dimension (1 for grayscale)
    y = np.array(data_depth)  # Depths

    return X, y


