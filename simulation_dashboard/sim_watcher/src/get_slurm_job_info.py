# =============================================================================
# SECTION: Imports
# =============================================================================


from datetime import datetime, timedelta
import subprocess
import os
import pandas as pd

import plasma_fusion_simulation_tools.simulation_dashboard.sim_watcher.default_values as defval


# =============================================================================
# SECTION: Main slurm job info generation
# =============================================================================

def get_slurm_job_df_info(username:str=defval.username, 
                          end_date=None, start_date=None, verbose:bool = False):
    
    job_csv_info_path = os.path.join(defval.default_job_csv_path, defval.default_job_csv_name)
    
    if os.path.exists(job_csv_info_path):
        if verbose: 
            print(f'Fetching data from: {job_csv_info_path}')
        
        slurm_jobs_df = load_existing_job_info()

        # slurm_jobs_df['Submit'] = pd.to_datetime(slurm_jobs_df['Submit'])
        csv_latest_date = slurm_jobs_df[['Submit', 'Start', 'End']].max().max()
        csv_earliest_date = slurm_jobs_df[['Submit', 'Start', 'End']].min().min()

        # csv_latest_date = slurm_jobs_df['Submit'].max()
        # csv_earliest_date = slurm_jobs_df['Submit'].min()

        # Default to current date if end_date is None, and set start_date to 30 days before end_date if it's None
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Convert to datetime if not already
        end_date = pd.to_datetime(end_date)
        start_date = pd.to_datetime(start_date)

        data_after_csv = pd.DataFrame()
        data_before_csv = pd.DataFrame()

        # Fetch data before CSV start date if needed
        if end_date > csv_latest_date:
            if verbose:
                print("Fetching additional data after CSV's latest record.")
            data_after_csv = terminal_slurm_job_df(
                username, end_date=end_date, start_date=csv_latest_date, verbose=verbose)
        
        # Fetch data after CSV end date if needed
        if start_date < csv_earliest_date:
            if verbose:
                print("Fetching additional data before CSV's earliest record.")
            data_before_csv = terminal_slurm_job_df(
                username, end_date=csv_earliest_date, start_date=start_date, verbose=verbose)
        
        # Combine all data parts
        # slurm_jobs_df = pd.concat([data_after_csv, slurm_jobs_df, data_before_csv], ignore_index=True).sort_values(by='Submit', ascending=True)
        slurm_jobs_df = pd.concat([data_after_csv, slurm_jobs_df, data_before_csv], ignore_index=True)

    else:
        if verbose:
            print(f"No CSV file found at {job_csv_info_path}. Fetching new data.")
        slurm_jobs_df = terminal_slurm_job_df(username, end_date=end_date, start_date=start_date, verbose=verbose)

    
    # Check if the DataFrame is empty
    if slurm_jobs_df.empty:
        print("No data was found in the given time range. Consider a broader time range")
        return


    filtered_by_submit = slurm_jobs_df[(slurm_jobs_df['Submit'] >= start_date) & (slurm_jobs_df['Submit'] <= end_date)]
    cleaned_slurm_jobs_df = filtered_by_submit.drop_duplicates(subset='JobID', keep='first')
    slurm_jobs_df_sorted = cleaned_slurm_jobs_df.sort_values(by='Submit', ascending=False)

    # Save the combined dataset back to CSV for future use
    save_existing_job_info(slurm_jobs_df_sorted, verbose=verbose)

    return slurm_jobs_df_sorted





# =============================================================================
# SECTION: Terminal commands for getting slurm job info
# =============================================================================

def terminal_slurm_job_df(username:str=defval.username, end_date=None, start_date=None,
                            day_chunk_size:int=30, verbose:bool=False):

    # Make sure day_chunk_size is bound between [1,30]
    day_chunk_size = max(min(day_chunk_size, 30), 1)

    # If no input end and start dates for slurm job search are given default to TODAY and (TODAY - 30 days)
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=day_chunk_size)
    
    # Convert dates to start of said day
    end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if end_date < start_date:
        raise ValueError(f'Please ensure the end date ({end_date}) is LATER than the start date ({start_date}).')

    total_days_back = (end_date - start_date).days
    # Choose smaller value between days back or day_chunk_size (if only needing 3 days back we don't need to look 30 days back)
    day_chunk_size = min(day_chunk_size, total_days_back)

    if verbose:
        print(f'Start date: {start_date} /// End date: {end_date} /// Day Chunk Size: {day_chunk_size}')


    all_data = []
    format_str = "JobID,Submit,Start,End,NNodes,QOS,State,Elapsed,User"
    chunked_end_date = end_date
    while total_days_back > 0:
        
        # Determine the start of the chunked days (i.e. 30 days back from end date) - this will update as we step along
        chunked_start_date = chunked_end_date - timedelta(days=day_chunk_size)

        chunked_start_time = chunked_start_date.strftime("%Y-%m-%d")
        chunked_end_time = chunked_end_date.strftime("%Y-%m-%d")

        # Run terminal command to get job data
        base_cmd = f"sacct -u {username} --starttime={chunked_start_time} --endtime={chunked_end_time} -X --format={format_str} --parsable2"
        output = run_terminal_command(base_cmd)

        lines = output.split('\n')[2:]  # Skip the header lines
        data = [line.split('|') for line in lines if line]  # Skip empty lines
        all_data.extend(data)

        # Update dates for the next chunk
        chunked_end_date = chunked_start_date - timedelta(days=1)  # Move end_date back to the day before the current chunk's start
        total_days_back -= day_chunk_size

        if verbose:
            print(base_cmd)
            add_entries = len(all_data)
            print(f'{add_entries} entries in the slurm job list.')

    columns = format_str.split(',')
    slurm_jobs_df = pd.DataFrame(all_data, columns=columns)

    modded_slurm_jobs_df = convert_df_columns(slurm_jobs_df)
    slurm_jobs_df_sorted = modded_slurm_jobs_df.sort_values(by='Submit', ascending=False)

    return slurm_jobs_df_sorted




def run_terminal_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error executing command '{cmd}': {error.decode('utf-8')}")

    return output.decode('utf-8')




# =============================================================================
# SECTION: Import and export slurm df data into CSVs
# =============================================================================

def save_existing_job_info(slurm_jobs_df, verbose=False,
                           filepath=defval.default_job_csv_path, 
                           filename=defval.default_job_csv_name):
    
    complete_path = os.path.join(filepath, filename)
    copy_slurm_jobs_df = slurm_jobs_df.copy()

    if os.path.exists(complete_path):
        existing_job_df = load_existing_job_info()
        
        combined_df = pd.concat([existing_job_df, copy_slurm_jobs_df], ignore_index=True)
        combined_df.drop_duplicates(subset='JobID', keep='first', inplace=True)
        combined_df.sort_values(by='Submit', ascending=False, inplace=True)
    else:
        copy_slurm_jobs_df.sort_values(by='Submit', ascending=False, inplace=True)
        combined_df = copy_slurm_jobs_df

    combined_df.to_csv(complete_path, index=False)
    
    if verbose:
        print(f"Data saved to {complete_path}")



def load_existing_job_info(filepath=defval.default_job_csv_path, 
                           filename=defval.default_job_csv_name):
    # Construct the complete path
    complete_path = os.path.join(filepath, filename)
    
    # Check if the complete file path exists
    if not os.path.exists(complete_path):
        raise FileNotFoundError(f'The file does not exist: {complete_path}\n Please specify a correct pathway in "default_values.py" or in the filepath variable.')
        
    # Attempt to load the CSV into a DataFrame
    try:
        csv_job_df = pd.read_csv(complete_path)
        modded_csv_job_df = convert_df_columns(csv_job_df)
        
        return modded_csv_job_df
    
    except FileNotFoundError as e:
        # You could log this error or handle it differently depending on your application's needs
        print(f"Error loading the CSV: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error
    


# =============================================================================
# SECTION: Helpers for dataframe management
# =============================================================================

def convert_df_columns(input_df):
    copy_input_df = input_df.copy()

    # Convert datetime columns
    datetime_columns = ['Submit', 'Start', 'End']
    for col in datetime_columns:
        if col in copy_input_df.columns:
            copy_input_df[col] = pd.to_datetime(copy_input_df[col], errors='coerce')
    
    # Convert 'JobID' to integer, with error handling
    if 'JobID' in copy_input_df.columns:
        # Use errors='coerce' to set invalid parsing as NaN, then fillna(0) or another value as needed
        copy_input_df['JobID'] = pd.to_numeric(copy_input_df['JobID'], errors='coerce').fillna(0).astype(int)

    return copy_input_df

