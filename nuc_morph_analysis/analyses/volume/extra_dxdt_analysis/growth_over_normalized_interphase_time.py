#%% 
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from nuc_morph_analysis.lib.preprocessing import add_times
import matplotlib
import matplotlib.pyplot as plt
from nuc_morph_analysis.analyses.volume.plot_help import group_and_extract, plot_dfg

# %%
# set up plot parameters and figure saving directory
matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({"font.size": 20})
plt.rcParams["figure.figsize"] = [5, 4]
figdir = f"volume/figures/local_growth/"

# %%
# load data
df = global_dataset_filtering.load_dataset_with_features()
df_full = filter_data.all_timepoints_full_tracks(df)
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

#%%
# now collect the average transient growth rates for different cell cycle windows

plot_type = 'mean'
colony_list = ['small','medium','large']
# plot the mean of the value (ycol) binned by xcol for given cell cycle windows
ycol = 'dxdt_48_volume'
xcol1 = 'dig_time'
dflist =[]
for colony in colony_list:

    fig,curr_ax = plt.subplots(1,1,figsize=(5,4))
    dfc = df_full[df_full['colony']==colony]
        

    curr_ax = plot_dfg(dfc,xcol1,ycol,f"{colony.capitalize()}",curr_ax,plot_type=plot_type,colorby='colony')
    
    plt.show()
