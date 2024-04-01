import numpy as np
from typing import Tuple, Dict




def compute_delta_angles(complex_num_array: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, float]]:
    # Compute the vector differences between consecutive complex numbers to get direction vectors
    vectors = np.diff(complex_num_array)

    angles = []
    for v1, v2 in zip(vectors[:-1], vectors[1:]):
        if np.isclose(np.abs(v1), 0) or np.isclose(np.abs(v2), 0):
            angles.append(0)
        else:
            angle = np.angle(v1/v2)
            angles.append(angle)

    angles = np.array(angles)
    angles = np.abs((angles + np.pi) % (2 * np.pi) - np.pi)
    
    total_samples = len(angles)

    average_angle = np.mean(angles)
    std_angle = np.std(angles, ddof=1)
    rel_high_angle_count = np.sum(angles > np.pi/2) / total_samples

    angle_stats = {'ave': average_angle, 'std': std_angle}

    return angles, rel_high_angle_count, angle_stats




def compute_fourier_decomposition(complex_num_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    # Compute the Fourier transform
    fft_result = np.fft.fft(complex_num_array)

    # Compute the frequency axis and normalize it
    freq = np.abs(np.fft.fftfreq(len(complex_num_array)))
    norm_freq = freq / np.max(np.abs(freq))

    # Compute the magnitude and normalize it
    magnitude = np.abs(fft_result)
    norm_mag = magnitude / np.sum(magnitude)

    # Calculate weighted average frequency and its standard deviation
    average_freq = np.sum(norm_freq * norm_mag)
    variance_freq = np.sum((norm_freq - average_freq) ** 2 * norm_mag)
    std_freq = np.sqrt(variance_freq)

    fft_stats = {'ave': average_freq, 'std': std_freq}

    return norm_freq, norm_mag, fft_stats
