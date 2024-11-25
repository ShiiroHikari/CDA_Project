#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:20:37 2024

@author: Basile Dupont
"""


import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.signal import butter, filtfilt, hilbert
import data_generation.arrival_time

# Generate signal with discrete exponential tails and random radiation in the coda
def generate_diracs(delta_pP, delta_sP, dt=0.1, duration=60, tau=3.0, coda_duration=random.uniform(20, 60), plot=False):
    """
    Generates a signal with Diracs for P, pP, and sP, each followed by an exponential tail 
    represented by a series of Diracs, with random sign flips in the coda.
    
    Parameters:
    - delta_pP: delay pP-P in seconds
    - delta_sP: delay sP-P in seconds
    - dt: sampling step (in seconds); 10 Hz
    - duration: total signal duration (in seconds)
    - tau: time constant for exponential decay (in seconds)
    - coda_duration: maximum duration of the tail (in seconds)
    - flip_probability: probability of flipping the sign in the coda
    
    Returns:
    - signal: array containing the signal
    - time: array containing the corresponding time instances
    """
    # Discrete time
    time = np.arange(0, duration, dt)
    signal = np.zeros_like(time)
    
    # Function to add an exponential tail as Diracs with random flips
    def add_coda_diracs(signal, start_index, amplitude, initial_sign):
        sign = initial_sign
        for i in range(1, int(coda_duration / dt)):
            index = start_index + i
            if index >= len(signal):  # If exceeding signal duration
                break
                
            # Randomly flip the sign
            sign = random.choice([-1, 1])
            signal[index] += sign * amplitude * np.exp(-i * dt / tau)
    
    # Amplitude and radiation (random sign) of the P wave
    amplitude_P = random.uniform(0.5, 1.0)  # Amplitude between 0.5 and 1
    sign_P = random.choice([-1, 1])  # Random sign for P to simulate radiation pattern at source
    signal[0] = sign_P * amplitude_P  # Simulate P-wave dirac
    add_coda_diracs(signal, 0, amplitude_P, sign_P)  # Add tail to simulate exponential energy decrease
    
    # Position of Diracs and tails
    pP_index = int(delta_pP / dt)
    sP_index = int(delta_sP / dt)
    
    # pP
    if pP_index < len(signal):
        amplitude_pP = amplitude_P * random.uniform(0, 1.1)  # Amplitude between 0% and 110% of P
        sign_pP = random.choice([-1, 1])  # Random sign for pP to simulate radiation pattern at source
        signal[pP_index] = sign_pP * amplitude_pP
        add_coda_diracs(signal, pP_index, amplitude_pP, sign_pP) # Add tail to simulate exponential energy decrease
    
    # sP
    if sP_index < len(signal):
        amplitude_sP = amplitude_P * random.uniform(0, 1.1)  # Amplitude between 0% and 110% of P
        sign_sP = random.choice([-1, 1])  # Random sign for sP to simulate radiation pattern at source
        signal[sP_index] = sign_sP * amplitude_sP
        add_coda_diracs(signal, sP_index, amplitude_sP, sign_sP) # Add tail to simulate exponential energy decrease

    if plot is True:
        plt.figure(figsize=(12,5))
        plt.stem(time, signal, basefmt=" ", label="Raw Signal")
        plt.title("Raw Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    return signal, time


# Convolution of the signal with a wavelet
def generate_ricker_wavelet(f_c=1.65, dt=0.1, length=1.0):
    """
    Generates a Ricker wavelet.
    
    Parameters:
    - f_c: central frequency (Hz) ; center of 0.8-2.5 later filter
    - dt: sampling step (s); 10 Hz
    - length: total duration of the wavelet (s)
    
    Returns:
    - w: array containing the wavelet
    - t: array of corresponding times
    """
    t = np.arange(-length / 2, length / 2, dt)
    w = (1 - 2 * (np.pi * f_c * t)**2) * np.exp(-(np.pi * f_c * t)**2)
    w /= np.sum(np.abs(w)) # Normalize the wavelet to avoid amplification
    
    return w, t


def convolve_signal_with_wavelet(signal, time, plot=False):
    """
    Convolves a discrete signal with a wavelet.
    
    Parameters:
    - signal: input signal
    - wavelet: wavelet for convolution
    
    Returns:
    - signal_convolved: convolved signal
    """
    wavelet, wavelet_time = generate_ricker_wavelet()
    signal_convolved = np.convolve(signal, wavelet, mode="same")
    
    if plot is True:
        plt.figure(figsize=(12, 6))
        
        # Wavelet
        plt.subplot(2,1,1)
        plt.plot(wavelet_time, wavelet, label="Ricker Wavelet")
        plt.title("Ricker Wavelet")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
        
        # Convolved Signal
        plt.subplot(2,1,2)
        plt.plot(time, signal_convolved, label="Convolved Signal")
        plt.title("Signal Convolved with Wavelet")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    return signal_convolved


# Add gaussian white noise 
def add_white_noise(signal, snr_db=random.uniform(2, 5)):
    """
    Adds Gaussian white noise to a signal based on the specified signal-to-noise ratio (SNR).
    
    Parameters:
    - signal : array containing the original signal
    - snr_db : signal-to-noise ratio in decibels (dB)
    
    Returns:
    - noisy_signal : the signal with added noise
    """
    # Signal power
    signal_power = np.mean(signal**2)
    # Noise power to achieve the desired SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    # Generate the noise
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    # Add noise to the signal
    noisy_signal = signal + noise
    
    return noisy_signal, snr_db


# Bandpass filter
def bandpass_filter(signal, lowcut=0.8, highcut=2.5, fs=100, order=3):
    """
    Applies a Butterworth bandpass filter.
    
    Parameters:
    - signal : array containing the signal
    - lowcut : lower cutoff frequency (Hz)
    - highcut : upper cutoff frequency (Hz)
    - fs : sampling frequency (Hz)
    - order : filter order (default is 4)
    
    Returns:
    - filtered_signal : filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')  # Filter coefficients
    filtered_signal = filtfilt(b, a, signal)  # Apply the filter

    # Z-score normalization
    filtered_signal_normalized = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    
    return filtered_signal_normalized

# Hilbert envelope extraction
def extract_hilbert_envelope(signal):
    """
    Extracts the analytic envelope of a signal using the Hilbert transform.
    
    Parameters:
    - signal : array containing the signal
    
    Returns:
    - envelope : analytic envelope of the signal
    """
    analytic_signal = hilbert(signal)  # Analytic signal
    envelope = np.abs(analytic_signal)  # Envelope (magnitude of the analytic signal)
    envelope /= np.max(envelope)  # Normalize to 1
    
    return envelope



# Generate signal from delta_pP and delta_sP
def generate_one_signal(plot=False):
    # Generate arrival times
    deltas, source, station = data_generation.arrival_time.generate_arrival_samples(num_stations=1)
    delta_pP, delta_sP = deltas[0][0], deltas[0][1]
    
    # Generate diracs
    diracs, time = generate_diracs(delta_pP, delta_sP)

    # Generate signal
    signal = convolve_signal_with_wavelet(diracs, time)

    # Add noise
    noisy_signal, snr_db = add_white_noise(signal)

    # Filter signal
    filtered_signal = bandpass_filter(noisy_signal)

    # Get Hilbert enveloppe
    envelope = extract_hilbert_envelope(filtered_signal)

    if plot is True:
        plt.figure(figsize=(15, 15))

        plt.subplot(511)
        plt.stem(time, diracs, markerfmt=' ', basefmt=' ')
        plt.vlines(delta_pP, min(diracs), max(diracs), color='r')
        plt.vlines(delta_sP, min(diracs), max(diracs), color='r')
        plt.title("Diracs")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        
        plt.subplot(512)
        plt.plot(time, signal)
        plt.vlines(delta_pP, min(signal), max(signal), color='r')
        plt.vlines(delta_sP, min(signal), max(signal), color='r')
        plt.title("Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()

        plt.subplot(513)
        plt.plot(time, noisy_signal)
        plt.vlines(delta_pP, min(noisy_signal), max(noisy_signal), color='r')
        plt.vlines(delta_sP, min(noisy_signal), max(noisy_signal), color='r')
        plt.title(f"Noisy signal (SNR = {snr_db:.1f} dB)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()

        plt.subplot(514)
        plt.plot(time, filtered_signal)
        plt.vlines(delta_pP, min(filtered_signal), max(filtered_signal), color='r')
        plt.vlines(delta_sP, min(filtered_signal), max(filtered_signal), color='r')
        plt.title("Filtered signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()

        plt.subplot(515)
        plt.plot(time, envelope)
        plt.vlines(delta_pP, min(envelope), max(envelope), color='r')
        plt.vlines(delta_sP, min(envelope), max(envelope), color='r')
        plt.title("Normalized Hilbert envelope")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        
        plt.tight_layout()
        plt.show()
        
    return envelope, source, station
    
    
# Generate signal from delta_pP and delta_sP for multiple stations
def generate_signals(num_stations=50):
    """
    Generate signals for multiple stations given a single source.
    
    Parameters:
    - num_stations : number of stations to simulate (default is 50)
    
    Returns:
    - results : list of tuples (envelope, source, station) for each station
    """
    # Generate arrival times for multiple stations
    deltas, source, stations = data_generation.arrival_time.generate_arrival_samples(num_stations)
    
    results = []
    
    for i, (delta_pP, delta_sP) in enumerate(deltas):
        # Generate diracs
        diracs, time = generate_diracs(delta_pP, delta_sP)
        
        # Generate signal
        signal = convolve_signal_with_wavelet(diracs, time)
        
        # Add noise
        noisy_signal, snr_db = add_white_noise(signal)
        
        # Filter signal
        filtered_signal = bandpass_filter(noisy_signal)
        
        # Get Hilbert envelope
        envelope = extract_hilbert_envelope(filtered_signal)
        
        # Append results for this station
        results.append((envelope, source, stations[i]))
    
    return results


def generate_matrix(num_stations=50):
    results = generate_signals(num_stations=num_stations)

    # Get depth (same for all)
    depth = results[0][1][2]

    # Initialize signal matrix
    num_samples = len(results[0][0])  # Number of points per signal
    signal_matrix = np.zeros((num_stations, num_samples))
    
    # Build signal matrix
    for i, (envelope, source, station) in enumerate(results):
        # Add normalized envelope to matrix
        signal_matrix[i, :] = envelope

    return signal_matrix, depth