
import os
import re

from GENE_sim_tools.GENE_sim_reader.src.utils.file_functions import switch_suffix_file

def plot_mode_structure(input_filepath:str):


    # TODO - Move the below logic to a function
    if os.path.isfile(input_filepath):
        # Check if the file ends with '.dat' or matches the '_XXXX' pattern
        if input_filepath.endswith('.dat') or re.search(r'_\d{4}$', input_filepath):
            field_filepath = switch_suffix_file(input_filepath, 'field')
        else:
            FileNotFoundError('Input file must be a GENE associated file ending with a suffix (i.e. "0001" or ".dat")')
    elif os.path.isdir(input_filepath):
        
        if 'parameters' not in os.listdir(input_filepath):
            raise NotADirectoryError('Ensure the input filepath is a directory containing a parameters file.')

        suffix = check_suffix(input_filepath)

        field_filepath = os.path.join(input_filepath, 'field' + suffix)

        pass
    else:
        raise NotADirectoryError('Ensure filepath given is either a GENE file (i.e. path/to/omega_0001) or a directory.')
    # filepath 





def check_suffix(input_filepath:str):

    # Ask for suffix repeatedly until correct one is given and found (check that input suffix actually exists)



    #list suffix available to load
    files = os.listdir(input_filepath)

    suffix = '0001'

    return suffix