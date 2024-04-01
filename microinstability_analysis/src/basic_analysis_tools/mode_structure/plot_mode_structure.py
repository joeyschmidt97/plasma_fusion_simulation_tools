
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from GENE_sim_tools.GENE_sim_reader.src.dict_simulation_data import sim_filepath_to_df, criteria_smart_appender
from plasma_fusion_simulation_tools.microinstability_analysis.src.basic_analysis_tools.mode_structure.field_data_extraction import data_array_flattened, rescale_array
from plasma_fusion_simulation_tools.microinstability_analysis.src.basic_analysis_tools.mode_structure.mode_structure_resolution import compute_delta_angles, compute_fourier_decomposition





def plot_mode_structure(dir_filepath:str, input_criteria=['field_phi'], verbose:bool=False):

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

        _, rel_high_angle_count, _ = compute_delta_angles(complex_array)

        # if rel_high_angle_count > 0.25:

        print(rel_high_angle_count)

        plotting_function(complex_array, zgrid, field_name, verbose=verbose)






def output_mode_structure_data(dir_filepath:str, input_criteria=['field_phi'], verbose:bool=False):

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

        _, rel_high_angle_count, _ = compute_delta_angles(complex_array)

        if rel_high_angle_count > 0.25:
            resolution = 'UNRESOLVED'
        else:
            resolution = 'RESOLVED'

        return resolution












# #------------------------------------------------------------------------------------------------
# # Plotting tools for mode structure--------------------------------------------------------------
# #------------------------------------------------------------------------------------------------





def plotting_function(complex_array, zgrid, field_name, verbose:bool=False):

    if field_name == 'field_phi':
        title = r'$\phi$'
        real_label = r'$Re[\phi]$'
        imag_label = r'$Im[\phi]$'
        abs_label = r'$|\phi|$'
    elif field_name == 'field_apar':
        title = r'$A_{||}$'
        real_label = r'$Re[A_{||}]$'
        imag_label = r'$Im[A_{||}]$'
        abs_label = r'$|A_{||}|$'
    else:
        title = real_label = imag_label = abs_label = ''

    if verbose:
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        # Plot the complex field
        ax1.plot(zgrid, np.real(complex_array), label=real_label, color='red')
        ax1.plot(zgrid, np.imag(complex_array), label=imag_label, color='blue')
        ax1.plot(zgrid, np.abs(complex_array), label=abs_label, color='black')
        ax1.set_title(title)
        ax1.set_xlabel(r'$z/\pi$', size=18)
        ax1.legend()
        ax1.grid(True)

        # Plot angle difference histogram
        angles, _, _ = compute_delta_angles(complex_array)
        counts, bins = np.histogram(angles, bins=100)
        total_samples = len(angles)
        relative_frequency = counts / total_samples
        ax2.bar(bins[:-1], relative_frequency, width=np.diff(bins), edgecolor='black', align='edge')
        ax2.set_title('Histogram of Delta Angles (Relative to Sample Size)')
        ax2.set_xlabel('Delta Angle (rad)')
        ax2.set_ylabel('Relative Frequency')

        # Plot Fourier decomposition
        norm_freq, norm_mag, _ = compute_fourier_decomposition(complex_array)
        ax3.stem(norm_freq, norm_mag, 'b', markerfmt=" ", basefmt="-b")
        ax3.set_title('Normalized Magnitude Spectrum of the Fourier Transform')
        ax3.set_xlabel('Normalized Frequency')
        ax3.set_ylabel('Normalized Magnitude')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

    else:
        plt.figure(figsize=(8, 4))
        plt.title(title)
        plt.plot(zgrid, np.real(complex_array), label=real_label, color='red')
        plt.plot(zgrid, np.imag(complex_array), label=imag_label, color='blue')
        plt.plot(zgrid, np.abs(complex_array), label=abs_label, color='black')
        plt.xlabel(r'$z/\pi$', size=18)
        plt.legend()
        plt.grid(True)
        plt.show()



# #------------------------------------------------------------------------------------------------
# # Additional plotting tools----------------------------------------------------------------------
# #------------------------------------------------------------------------------------------------





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


