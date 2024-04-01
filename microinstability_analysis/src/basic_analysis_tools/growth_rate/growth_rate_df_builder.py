# import pandas as pd

<<<<<<< HEAD
from GENE_sim_tools.GENE_sim_reader.src.dict_simulation_data import sim_filepath_to_df, criteria_smart_appender
=======
from GENE_sim_tools.GENE_sim_reader.src.dict_simulation_data import sim_filepath_to_df
>>>>>>> a04de6c3267648088ca1d6870756055fbd482b67
from GENE_sim_tools.GENE_sim_reader.src.utils.file_functions import string_to_list

def growth_freq_dataframe(filepath, input_criteria = ['gamma', 'omega', 'status==CONVERGED']):

    mod_input_crit = criteria_smart_appender(input_criteria, ['gamma', 'omega', 'status==CONVERGED'])
    sim_df = sim_filepath_to_df(filepath_list=filepath, criteria_list=mod_input_crit)

    return sim_df
    





