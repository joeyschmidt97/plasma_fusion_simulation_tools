
# import os
# import re

# import numpy as np
# import matplotlib.pyplot as plt

# from GENE_sim_tools.GENE_sim_reader.src.dict_simulation_data import sim_filepath_to_df, criteria_smart_appender
# from plasma_fusion_simulation_tools.microinstability_analysis.src.basic_analysis_tools.mode_structure.plot_mode_structure import check_suffix
# from plasma_fusion_simulation_tools.microinstability_analysis.src.basic_analysis_tools.mode_structure.field_data_extraction import data_array_flattened, rescale_array




# def fourier_mode_structure(dir_filepath:str, input_criteria=['field_phi']):

#     suffix_crit_present = any('suffix' in crit for crit in input_criteria)
#     if not suffix_crit_present:
#         suffix = check_suffix(dir_filepath)
#         suffix_crit = f'suffix=={suffix}'
#         input_criteria.append(suffix_crit)

#     mod_criteria = criteria_smart_appender(input_criteria, ['field_phi', 'time==last'])    
#     sim_df = sim_filepath_to_df(dir_filepath, criteria_list=mod_criteria)


#     field_list = []
#     if any('field_phi' in crit for crit in mod_criteria):
#         field_list.append('field_phi')
#     if any('field_apar' in crit for crit in mod_criteria):
#         field_list.append('field_apar')

#     for field_name in field_list:
#         flat_array, zgrid = data_array_flattened(sim_df, field_name)
#         complex_array = rescale_array(flat_array, sim_df)
