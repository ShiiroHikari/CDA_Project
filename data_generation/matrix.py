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
def generate_matrix(num_stations=50, depth=None, rand_inactive=0, use_TauP=False):
    """
    Generates a matrix and depth associated.
    
    Parameters:
    - num_stations : number of stations per depth
    - depth : depth to simulate (default is None)
    - rand_inactive : max number of inactive stations
    - use_TauP : whether to use or not TauP model for propagation
    
    Returns:
    - signal_matrix : matrix with one line per signal
    - depth : the depth corresponding to this matrix
    """
    results, distances = data_generation.signal.generate_signals(num_stations=num_stations, depth=depth, rand_inactive=rand_inactive, use_TauP=use_TauP)

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
    Normalizes distances using min-max scaling, with 0 values (inactive stations) set to -1.

    Parameters:
    - distances : array of shape [batch_size, num_stations], distances in meters.
                  Inactive stations must be marked as 0.
    - min_distance : Theoretical minimum distance in meters (default is 2.5e6).
    - max_distance : Theoretical maximum distance in meters (default is 1e7).

    Returns:
    - Normalized distances of shape [batch_size, num_stations], with inactive stations set to -1.
    """
    # Min-max scaling for positive distances (active stations)
    normalized = (distances - min_distance) / (max_distance - min_distance)
    
    # Replace inactive stations (originally 0) with -1
    normalized[distances == 0] = -1
    
    return normalized



# Generate multiple matrix for model training
def dataset_generation(num_entries=32, num_stations=50, depth_list=None, rand_inactive=0, use_TauP=False):
    """
    Generates a dataset containing signal matrices and their corresponding depths.
    
    Parameters:
    - num_entries : number of cases (different depths) to generate
    - num_stations : number of stations per depth
    - depth_list : list of depths to generate (should have num_entries size or None)
    - rand_inactive : max number of inactive stations
    - use_TauP : whether to use or not TauP model for propagation
    
    Returns:
    - X : numpy array of shape (num_entries, 1, num_stations, X) (signal matrices)
    - y : numpy array of shape (num_entries,) (depths)
    """
    data_matrix = []
    data_depth = [] 
    data_distances = []
    
    for i in range(num_entries):
        # Generate signal matrix and depth
        signal_matrix, depth, distances = generate_matrix(num_stations=num_stations, depth=depth_list[i] if depth_list is not None else None, rand_inactive=rand_inactive, use_TauP=use_TauP)
        
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


