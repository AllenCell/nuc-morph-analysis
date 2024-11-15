# %% SuppFigS10 E
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.volume.plot_help import update_plotting_params, adjust_axis_positions, plot_dip_detection_validation
from nuc_morph_analysis.lib.preprocessing import filter_data
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
import numpy as np
#%%
#%% update plotting parameters
fs, fw, fh = update_plotting_params()

#%%
# load the data
df0 = load_dataset_with_features('all_baseline',load_local=True, remove_growth_outliers=False)
df0 = filter_data.all_timepoints_minimal_filtering(df0)
df_full = filter_data.all_timepoints_full_tracks(df0)

#%%
save_dir = Path(__file__).parent / 'figures' / 'volume_dip_filter_assessment'

#%%

track_list = [7875,72414,86246,85345,77291,86570,83675,84663,86212]
#%%
from nuc_morph_analysis.analyses.volume import filter_out_dips
find_dips=True
dftracks = df_full[df_full['track_id'].isin(track_list)]
columns_to_remove = dftracks.columns[dftracks.columns.str.contains('dip|jump')]
dftracks = dftracks.drop(columns=columns_to_remove)


dftracks_out = filter_out_dips.run_script(dftracks,
                                          return_intermediates=False,
                                          use_detrended=True,)
    

   #%%
for track in track_list:
    plot_dip_detection_validation(dftracks_out[dftracks_out['track_id'] == track],peak_str = 'dips')
    

