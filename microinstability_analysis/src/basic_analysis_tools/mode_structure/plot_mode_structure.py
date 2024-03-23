
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from GENE_sim_tools.GENE_sim_reader.src.dict_simulation_data import sim_filepath_to_df, criteria_smart_appender
from plasma_fusion_simulation_tools.microinstability_analysis.src.basic_analysis_tools.mode_structure.field_data_extraction import data_array_flattened, rescale_array




def plot_mode_structure(dir_filepath:str, input_criteria=['field_phi']):

    suffix_crit_present = any('suffix' in crit for crit in input_criteria)
    if not suffix_crit_present:
        suffix = check_suffix(dir_filepath)
        suffix_crit = f'suffix=={suffix}'
        input_criteria.append(suffix_crit)

    mod_criteria = criteria_smart_appender(input_criteria, ['field_phi', 'time==last'])    
    sim_df = sim_filepath_to_df(dir_filepath, criteria_list=mod_criteria)


    field_list = []
    if any('field_phi' in crit for crit in mod_criteria):
        field_list.append('field_phi')
    if any('field_apar' in crit for crit in mod_criteria):
        field_list.append('field_apar')

    for field_name in field_list:
        flat_array, zgrid = data_array_flattened(sim_df, field_name)
        complex_array = rescale_array(flat_array, sim_df)

        _, rel_high_angle_count = compute_delta_angles(complex_array)
        if rel_high_angle_count > 0.25:

            plotting_function(complex_array, zgrid, field_name)
            fourier_decomposition(complex_array)
            plot_angle_diff_histogram(complex_array)









def compute_delta_angles(complex_num_array):

    # Compute the vector differences between consecutive complex numbers to get direction vectors
    vectors = np.diff(complex_num_array)

    # Ensure the calculation of angles does not lead to NaN values
    # This might happen if the vectors have zero length or are too close to each other
    angles = []
    for v1, v2 in zip(vectors[:-1], vectors[1:]):
        if np.isclose(np.abs(v1), 0) or np.isclose(np.abs(v2), 0):
            angles.append(0)  # Assign 0 or a small angle if the vector length is close to 0
        else:
            angle = np.angle(v1/v2)
            angles.append(angle)

    # Convert the list of angles to a numpy array
    angles = np.array(angles)

    # Ensure angle differences are within the range [-pi, pi]
    angles = np.abs((angles + np.pi) % (2 * np.pi) - np.pi)
    
    # # Calculate the histogram with normalization to the total number of samples
    # counts, bins = np.histogram(angles, bins=10)
    total_samples = len(angles)
    # relative_frequency = counts / total_samples  # Calculate relative frequency

    rel_high_angle_count = np.sum(angles > np.pi/2)/total_samples

    return angles, rel_high_angle_count



def plot_angle_diff_histogram(complex_num_array):

    angles, rel_high_angle_count = compute_delta_angles(complex_num_array)

    counts, bins = np.histogram(angles, bins=10)
    total_samples = len(angles)
    relative_frequency = counts / total_samples  # Calculate relative frequency

    print('UNRESOLVED SIMULATION')
    print(rel_high_angle_count)

    # Plot the histogram with frequencies adjusted to the sample size
    plt.bar(bins[:-1], relative_frequency, width=np.diff(bins), edgecolor='black', align='edge')
    plt.title('Histogram of Delta Angles (Relative to Sample Size)')
    plt.xlabel('Delta Angle (rad)')
    plt.ylabel('Relative Frequency')
    plt.show()











def fourier_decomposition(complex_num_array):
    # Compute the Fourier transform
    fft_result = np.fft.fft(complex_num_array)

    # Compute the frequency axis and take the absolute to handle negative frequencies
    freq = np.abs(np.fft.fftfreq(len(complex_num_array)))
    norm_freq = freq / np.max(freq)

    # Compute the magnitude
    magnitude = np.abs(fft_result)
    normalized_magnitude = magnitude / np.sum(magnitude)

    # Sum of normalized magnitude for frequencies above 0.25
    sum_high_freq_magnitude = np.sum(normalized_magnitude[norm_freq > 0.25])
    print("Sum of normalized magnitude for freq > 0.25:", sum_high_freq_magnitude)

    # Plot the Fourier transform (magnitude spectrum)
    plt.figure(figsize=(8, 4))
    plt.stem(norm_freq, normalized_magnitude, 'b', markerfmt=" ", basefmt="-b")
    plt.title('Normalized Magnitude Spectrum of the Fourier Transform')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Normalized Magnitude')
    plt.grid(True)
    plt.show()



# #------------------------------------------------------------------------------------------------
# # Plotting tools for mode structure--------------------------------------------------------------
# #------------------------------------------------------------------------------------------------


def plotting_function(complex_array, zgrid, field_name):

    if field_name=='field_phi':
        title=r'$\phi$'
        real_label=r'$Re[\phi]$'
        imag_label=r'$Im[\phi]$'
        abs_label=r'$|\phi|$'

    elif field_name=='field_apar':
        title=r'$A_{||}$'
        real_label=r'$Re[A_{||}]$'
        imag_label=r'$Im[A_{||}]$'
        abs_label=r'$|A_{||}|$'

    else:
        pass

    # Plotting the real part of the complex numbers
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.plot(zgrid, np.real(complex_array), label=real_label, color='red')
    plt.plot(zgrid, np.imag(complex_array), label=imag_label, color='blue')
    plt.plot(zgrid, np.abs(complex_array), label =abs_label, color='black')
    plt.xlabel(r'$z/\pi$',size=18)
    plt.legend()
    plt.grid(True)
    plt.show()




# #------------------------------------------------------------------------------------------------
# # Fetch suffix data from current directory-------------------------------------------------------
# #------------------------------------------------------------------------------------------------

def check_suffix(input_filepath:str):
    suffix_options = set()
    #list suffix available to load
    files = os.listdir(input_filepath)
    for file in files:
        suffix_match = re.search(r'_\d{4}$', file)
        
        if file.endswith('.dat'):
            end_suffix = '.dat'
        elif suffix_match:
            end_suffix = re.search(r'\d{4}$', file).group()
        else:
            pass

        suffix_options.add(end_suffix)

    suffix = input(f'Please choose a suffix from the list given (0001 for example - without quotes): \n {list(suffix_options)}')
    
    if suffix =='.dat':
        pass
    elif re.search(r'\d{4}$', suffix):
        pass
    else:
        raise ValueError(f'Please ensure "{suffix}" is either input as .dat or XXXX where X are integers (i.e. 0002)')

    return suffix


