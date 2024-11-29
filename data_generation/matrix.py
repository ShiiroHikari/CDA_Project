#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 23:57:18 2024

@author: Basile Dupont
"""


import numpy as np
import torch
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
    results, distances = data_generation.signal.generate_signals(num_stations=num_stations)

    # Get depth (same for all)
    depth = results[0][1][2] # from 1st sample

    # Initialize matrix
    num_samples = len(results[0][0])
    signal_matrix = np.zeros((num_stations, num_samples))
    
    # Build matrix
    for i, (envelope, _, _) in enumerate(results):
        signal_matrix[i, :] = envelope

    return signal_matrix, depth, distances



# Normalize distances
def normalize_distances(distances, min_distance, max_distance):
    """
    Min-Max scales distances using the theoretical range.

    Args:
        distances : shape [batch_size, num_stations] distances in meters.
        min_distance : Theoretical minimum distance in meters.
        max_distance : Theoretical maximum distance in meters.

    Returns:
        Min-max scaled distances, shape [batch_size, num_stations]
    """
    return (distances - min_distance) / (max_distance - min_distance)



# Generate multiple matrix for model training
def dataset_generation(num_entries=1000, num_stations=50):
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
    data_distances = []
    
    for i in range(num_entries):
        # Generate signal matrix and depth
        signal_matrix, depth, distances = generate_matrix(num_stations=num_stations)
        
        # Save matrix, depth and distances
        data_matrix.append(signal_matrix)
        data_depth.append(depth)
        data_distances.append(distances)

    # Get signal shape
    signal_shape = data_matrix[0].shape[1]

    # Convert to NumPy arrays
    data_matrix = np.array(data_matrix)  # Ensure data_matrix is a NumPy array
    data_depth = np.array(data_depth)  # Ensure data_depth is a NumPy array
    data_distances = np.array(data_distances)  # Ensure data_distances is a NumPy array

    # Convert NumPy arrays to PyTorch tensors
    X = torch.from_numpy(data_matrix).float().reshape(-1, 1, num_stations, signal_shape)  # Shape [num_entries, 1, num_stations, signal_shape]
    y = torch.from_numpy(data_depth).float().view(-1, 1)  # Shape [num_entries, 1] for depths
    D = torch.from_numpy(data_distances).float()  # Shape [num_entries, num_stations] for distances

    # Normalize distances
    min_D = 2.5e6 # Minimum distance is above 2.5 thousand km
    max_D = 1e7 # Maximum distance is under 10 thousand km
    D = normalize_distances(D, min_D, max_D)
    

    return X, y, D, signal_shape


