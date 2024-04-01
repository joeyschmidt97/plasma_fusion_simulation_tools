import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from plasma_fusion_simulation_tools.microinstability_analysis.src.basic_analysis_tools.growth_rate.growth_rate_df_builder import growth_freq_dataframe





def gamma_omega_plot(filepath, criteria_list = ['gamma', 'omega'], plot_mode='kymin', label_key='nz0', verbose = False):

    if isinstance(filepath, str): filepath = [filepath]

    gr_sim_df = growth_freq_dataframe(filepath, criteria_list) 
    gr_sim_df = gr_sim_df.sort_values(by='kymin')

    
    #TODO - Make this optional to avoid legend always being there is no label_key is given
    # Assuming gr_sim_df[label_key] is a categorical variable with unique values
    unique_values = gr_sim_df[label_key].unique()
    unique_values = np.sort(unique_values)
    colors = plt.cm.brg(np.linspace(0, 1, len(unique_values)))

            
    
    #TODO - default always plot this as there will always be at least 1 filepath
    base_df = gr_sim_df[gr_sim_df[label_key] == gr_sim_df[label_key].unique()[0]]

    unique_filepath_len = len(gr_sim_df['filepath'].unique())

    if unique_filepath_len == 1:
        fig, (gamma_ax, omega_ax) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    elif unique_filepath_len > 1:
        fig, ((gamma_ax, omega_ax), (gamma_diff, omega_diff)) = plt.subplots(nrows=2, ncols=2, 
                                                                             gridspec_kw={'height_ratios': [2, 1]}, 
                                                                             figsize=(10, 6), sharex='col')
        
        gamma_diff.set_xscale('log')
        gamma_diff.set_ylabel('% Diff (gamma)', fontsize=15)
        gamma_diff.grid(True)

        omega_diff.set_xscale('log')
        omega_diff.set_ylabel('% Diff (omega)', fontsize=15)
        omega_diff.grid(True)
                
        #Find min and max x-values for accurate percent diff comparison (Make filepath length dependent)
        min_x = max(gr_sim_df.groupby(label_key)['kymin'].min())
        max_x = min(gr_sim_df.groupby(label_key)['kymin'].max())

        # Now, when interpolating and plotting, limit the x-values to this range
        base_df = base_df[(base_df['kymin'] >= min_x) & (base_df['kymin'] <= max_x)]
        threshold = 0.01  # Define a threshold for small reference values




                
            
    # Iterate over unique values and plot with different colors
    for i, unique_value in enumerate(unique_values):
        subset_df = gr_sim_df[gr_sim_df[label_key] == unique_value]

        plot_label = f"{label_key} = {unique_value}"

        gamma_ax.set_xscale('log')
        gamma_ax.set_xlabel('kymin', fontsize=15)
        gamma_ax.set_ylabel('gamma (cs/a)', fontsize=15)
        gamma_ax.plot(subset_df['kymin'], subset_df['gamma'], label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
        gamma_ax.legend()

        omega_ax.set_xscale('log')
        omega_ax.set_xlabel('kymin', fontsize=15)
        omega_ax.set_ylabel('omega (cs/a)', fontsize=15)
        omega_ax.plot(subset_df['kymin'], subset_df['omega'], label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
        omega_ax.legend()


        if verbose:
            print('x-values for filepath:', subset_df['filepath'].unique())
            x_decimal_list = [format_value_as_int_or_float(value) for value in subset_df['kymin']]


            for _, (x_val, y_val) in enumerate(zip(subset_df['kymin'], subset_df['gamma'])):
                label = f"{format_value_as_int_or_float(x_val)}"  # Format the label with two decimal places
                gamma_ax.annotate(label, xy=(x_val, y_val), xytext=(-5, 10), textcoords='offset points')



        if i > 0:  # Skip the first unique value since it's used as reference
            # Interpolate gamma and omega for direct comparison with base_df
            interpolated_gamma = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['gamma'])
            interpolated_omega = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['omega'])

            # Initialize lists to store calculated differences and markers
            gamma_diff_values = []
            omega_diff_values = []
            markers = []

            # Calculate differences using hybrid method
            for base_gamma, comp_gamma, base_omega, comp_omega in zip(base_df['gamma'].values, interpolated_gamma, base_df['omega'].values, interpolated_omega):
                # print(base_gamma, threshold, abs(base_gamma) < threshold)
                
                if abs(base_gamma) < threshold:
                    gamma_diff_values.append(comp_gamma - base_gamma)
                    omega_diff_values.append(comp_omega - base_omega)
                    markers.append('x')  # Marker for absolute differences
                else:
                    gamma_diff_values.append((comp_gamma - base_gamma) / base_gamma * 100)
                    omega_diff_values.append((comp_omega - base_omega) / base_omega * 100)
                    markers.append('o')  # Marker for percent differences


            # Plot differences using different markers as necessary
            for kymin_val, gamma_diff_val, omega_diff_val, marker in zip(base_df['kymin'], gamma_diff_values, omega_diff_values, markers):
                gamma_diff.plot(kymin_val, gamma_diff_val, label=f"{plot_label} (Gamma)", marker=marker, linestyle='None', color=colors[i])
                omega_diff.plot(kymin_val, omega_diff_val, label=f"{plot_label} (Omega)", marker=marker, linestyle='None', color=colors[i])


    gamma_ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
    omega_ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)

    plt.tight_layout()
    plt.show()

    #     # Calculate and plot percent differences if not the base series
    #     if i > 0:  # Skip the first unique value since it's used as reference
    #         # Interpolate gamma and omega for direct comparison with base_df
    #         interpolated_gamma = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['gamma'])
    #         interpolated_omega = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['omega'])

    #         # Calculate percent differences
    #         gamma_percent_diff = ((interpolated_gamma - base_df['gamma'].values) / base_df['gamma'].values) * 100
    #         omega_percent_diff = ((interpolated_omega - base_df['omega'].values) / base_df['omega'].values) * 100

    #         # Plot percent differences using the same color and style for identification
    #         gamma_diff.plot(base_df['kymin'], gamma_percent_diff, label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
    #         omega_diff.plot(base_df['kymin'], omega_percent_diff, label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])


    # gamma_ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)
    # omega_ax.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True)

    # # Show the plot
    # plt.tight_layout()
    # plt.show()



                


                

def gamma_omega_plot_ARCHIVE(filepath, criteria_list = ['gamma', 'omega'], plot_mode='kymin', label_key='nz0', verbose = False):

    if isinstance(filepath, str): filepath = [filepath]

    gr_sim_df = growth_freq_dataframe(filepath, criteria_list) 
    
    converged = gr_sim_df['status'] == 'CONVERGED'
    gr_sim_df = gr_sim_df[converged]



    if label_key == 'bpar':
        if 'bpar' in gr_sim_df.columns:
            gr_sim_df['bpar'].fillna(False, inplace=True)
        else:
            gr_sim_df['bpar'] = False


    if (label_key.strip() == 'theta_0(deg)') or (label_key == 'theta_0'):
        gr_sim_df['kx_center'].fillna(0, inplace=True)
        gr_sim_df['theta_0'] = gr_sim_df['kx_center']/(gr_sim_df['shat']*gr_sim_df['kymin'])

        if label_key.strip() == 'theta_0(deg)':
            gr_sim_df['theta_0(deg)'] = gr_sim_df['theta_0']*(180/np.pi)
            gr_sim_df['theta_0(deg)'] = gr_sim_df['theta_0(deg)'].round(3)
            gr_sim_df['theta_0'] = gr_sim_df['theta_0'].round(3)
        else:
            gr_sim_df['theta_0'] = gr_sim_df['theta_0'].round(3)
        

        if plot_mode == 'theta_disk':
            unit_disk_theta = np.linspace(0, 2*np.pi, 100)
            plt.plot(np.cos(unit_disk_theta), np.sin(unit_disk_theta), color='black', linestyle='-', alpha=0.5)
            plt.scatter(np.cos(gr_sim_df['theta_0']), np.sin(gr_sim_df['theta_0']))
            plt.gca().set_aspect('equal')

            print(gr_sim_df['theta_0'].unique())
            

            # Iterate over unique values and annotate each point
            for theta_value in gr_sim_df['theta_0'].unique():
                x_val = np.cos(theta_value)
                y_val = np.sin(theta_value)

                label = f"{theta_value:.2f}"  # Format the label with two decimal places
                plt.annotate(label, xy=(x_val, y_val), xytext=(10, 0), textcoords='offset points')
            
            plt.show()



    gr_sim_df = gr_sim_df.sort_values(by='kymin')



    
    if len(filepath) == 1:
        fig, (gamma_ax, omega_ax) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    elif len(filepath) > 1:
        fig, ((gamma_ax, omega_ax), (gamma_diff, omega_diff)) = plt.subplots(nrows=2, ncols=2, 
                                                                             gridspec_kw={'height_ratios': [2, 1]}, 
                                                                             figsize=(10, 6), sharex='col')
    


        
    


    # Assuming gr_sim_df[label_key] is a categorical variable with unique values
    unique_values = gr_sim_df[label_key].unique()
    unique_values = np.sort(unique_values)

    # Create a colormap with a color for each unique value
    colors = plt.cm.brg(np.linspace(0, 1, len(unique_values)))
        


        
    # # Assuming this part is within your larger function that has already defined variables like gr_sim_df, etc.
    # base_df = gr_sim_df[gr_sim_df[label_key] == gr_sim_df[label_key].unique()[0]]  # Assuming the first unique value is the base

    # # Preparing the axes for percent difference plots if there are multiple file paths
    # if len(filepath) > 1:
    #     gamma_diff.set_xscale('log')
    #     gamma_diff.set_ylabel('% Difference for Gamma', fontsize=15)
    #     gamma_diff.grid(True)

    #     omega_diff.set_xscale('log')
    #     omega_diff.set_ylabel('% Difference for Omega', fontsize=15)
    #     omega_diff.grid(True)

    # for i, unique_value in enumerate(unique_values):
    #     subset_df = gr_sim_df[gr_sim_df[label_key] == unique_value]

    #     plot_label = f"{label_key} = {unique_value}"

    #     # Plot gamma and omega as before
    #     gamma_ax.plot(subset_df['kymin'], subset_df['gamma'], label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
    #     omega_ax.plot(subset_df['kymin'], subset_df['omega'], label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])

    #     # Calculate and plot percent differences if not the base series
    #     if i > 0:  # Skip the first unique value since it's used as reference
    #         # Interpolate gamma and omega for direct comparison with base_df
    #         interpolated_gamma = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['gamma'])
    #         interpolated_omega = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['omega'])

    #         # Calculate percent differences
    #         gamma_percent_diff = ((interpolated_gamma - base_df['gamma'].values) / base_df['gamma'].values) * 100
    #         omega_percent_diff = ((interpolated_omega - base_df['omega'].values) / base_df['omega'].values) * 100

    #         # Plot percent differences using the same color and style for identification
    #         gamma_diff.plot(base_df['kymin'], gamma_percent_diff, label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
    #         omega_diff.plot(base_df['kymin'], omega_percent_diff, label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])

    # # Adding legends outside the plots
    # gamma_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # omega_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # if len(filepath) > 1:
    #     gamma_diff.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    #     omega_diff.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

    # plt.tight_layout()
    # plt.show()


            
    
    # Assuming this part is within your larger function that has already defined variables like gr_sim_df, etc.
    base_df = gr_sim_df[gr_sim_df[label_key] == gr_sim_df[label_key].unique()[0]]  # Assuming the first unique value is the base

    # Preparing the axes for percent difference plots if there are multiple file paths
    if len(filepath) > 1:
        gamma_diff.set_xscale('log')
        gamma_diff.set_ylabel('% Difference for Gamma', fontsize=15)
        gamma_diff.grid(True)

        omega_diff.set_xscale('log')
        omega_diff.set_ylabel('% Difference for Omega', fontsize=15)
        omega_diff.grid(True)

        
            
    # Iterate over unique values and plot with different colors
    for i, unique_value in enumerate(unique_values):
        subset_df = gr_sim_df[gr_sim_df[label_key] == unique_value]

        plot_label = f"{label_key} = {unique_value}"

        gamma_ax.set_xscale('log')
        gamma_ax.set_xlabel('kymin', fontsize=15)
        gamma_ax.set_ylabel('gamma (cs/a)', fontsize=15)
        gamma_ax.plot(subset_df['kymin'], subset_df['gamma'], label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
        gamma_ax.legend()

        omega_ax.set_xscale('log')
        omega_ax.set_xlabel('kymin', fontsize=15)
        omega_ax.set_ylabel('omega (cs/a)', fontsize=15)
        omega_ax.plot(subset_df['kymin'], subset_df['omega'], label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
        omega_ax.legend()


        if verbose:
            print('x-values for filepath:', subset_df['filepath'].unique())
            x_decimal_list = [format_value_as_int_or_float(value) for value in subset_df['kymin']]



            for _, (x_val, y_val) in enumerate(zip(subset_df['kymin'], subset_df['gamma'])):
                label = f"{format_value_as_int_or_float(x_val)}"  # Format the label with two decimal places
                gamma_ax.annotate(label, xy=(x_val, y_val), xytext=(-5, 10), textcoords='offset points')



        # Calculate and plot percent differences if not the base series
        if i > 0:  # Skip the first unique value since it's used as reference
            # Interpolate gamma and omega for direct comparison with base_df
            interpolated_gamma = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['gamma'])
            interpolated_omega = np.interp(base_df['kymin'], subset_df['kymin'], subset_df['omega'])

            # Calculate percent differences
            gamma_percent_diff = ((interpolated_gamma - base_df['gamma'].values) / base_df['gamma'].values) * 100
            omega_percent_diff = ((interpolated_omega - base_df['omega'].values) / base_df['omega'].values) * 100

            # Plot percent differences using the same color and style for identification
            gamma_diff.plot(base_df['kymin'], gamma_percent_diff, label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])
            omega_diff.plot(base_df['kymin'], omega_percent_diff, label=plot_label, marker='o', markersize=5, linestyle='dotted', color=colors[i])



    # Show the plot
    plt.tight_layout()
    plt.show()






def format_value_as_int_or_float(value):
    """Converts a value to an int if it's equivalent; otherwise, keeps it as float without rounding.

    Args:
        value (float): The value to process.

    Returns:
        str: The value converted to a string, either as an int or a float.
    """
    try:
        # Attempt to convert to an integer if it doesn't change the value.
        if float(value) == int(float(value)):
            return int(value)
        else:
            # Value has significant decimal part, keep it as float without additional formatting.
            return float(value)
    except ValueError:
        # In case of ValueError, return the original value as string.
        # This branch might not be reached, but it's good practice to handle potential exceptions.
        return str(value)





