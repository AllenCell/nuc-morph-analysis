
# %%
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data 
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import add_times
from sklearn.linear_model import LinearRegression
import pandas as pd

#%%
df = load_dataset_with_features('all_baseline',load_local=True)
df_full = filter_data.all_timepoints_full_tracks(df)
#%%
# now validate behavior of digitize_time_column
add_times.validate_dig_time_with_plot()
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

colony_list = ['small','medium','large']


def add_mean_feature_over_trajectory_array_based(df0, feature_list, index_columns=['track_id','index_sequence','Fb','frame_transition'],mean_std='mean'):
    df = df0.dropna(subset=['frame_transition','Fb']) # extra NaN values in these columns can cause problems at log1 log2 steps
    log1 = df['index_sequence'] >= df['frame_transition'] # restrict track to start from frame_transition
    log2 = df['index_sequence'] <= df['Fb'] # restrict track to end at Fb (breakdown)
    log = log1 & log2
    df_mean = df.loc[log,list(set(index_columns+feature_list))] # only keep relevant columns


    # group by track_id and calculate mean
    if mean_std == 'mean':
        dfg = df_mean.groupby('track_id')[feature_list].mean()

        dfg.rename(columns={col: f'mean_{col}' for col in feature_list}, inplace=True)
    elif mean_std == 'std':
        dfg = df_mean.groupby('track_id')[feature_list].std()
        dfg.rename(columns={col: f'std_{col}' for col in feature_list}, inplace=True)
    
    # merge the results
    df2 = pd.merge(df0,dfg,on='track_id',suffixes=('','_dup2'),how='left')
    return df2


feature_list = ["volume_at_B","mean_volume","mean_height","mean_SA_vol_ratio","std_SA_vol_ratio","mean_mesh_sa","std_mesh_sa","mean_index_sequence","mean_colony_depth"]

df_full = add_mean_feature_over_trajectory_array_based(df_full,["volume","height","SA_vol_ratio","mesh_sa","index_sequence",'colony_depth'],mean_std='mean')
df_full = add_mean_feature_over_trajectory_array_based(df_full,["volume","height","SA_vol_ratio","mesh_sa","index_sequence",'colony_depth'],mean_std='std')

# drop duplicate columns

ycol = 'duration_BC'
for plot_type in ['mean']:
    # for xcol in ['height_avg','volume_avg','SA_vol_ratio_avg','index_sequence_avg','colony_depth_avg']:
    for xcol in feature_list:
        fig,curr_ax = plt.subplots(1,1,figsize=(3,2))
        for ci,colony in enumerate(colony_list):
            dfc = df_full[df_full['colony']==colony]

            # top column
            # dfg = group_and_extract(dfc,xcol1,ycol)
            dfg = dfc[['track_id',xcol,ycol]].groupby('track_id')
            dfgm = dfg.mean()
            dfgm.dropna(subset=[xcol,ycol],inplace=True)
            x = dfgm[xcol]
            y = dfgm[ycol]

            reg = LinearRegression().fit(x.values.reshape(-1,1),y.values)
            y_pred = reg.predict(x.values.reshape(-1,1))
            r2 = reg.score(x.values.reshape(-1,1),y.values)
            curr_ax.plot(x,y_pred,linestyle='--',label=f'{colony}, r2={r2:.2f}')
            assert type(curr_ax) == plt.Axes
            curr_ax.scatter(x,y,s=1)
            curr_ax.set_xlabel(xcol)
            curr_ax.set_ylabel(ycol)

        # now all
        dfc = df_full
        dfg = dfc[['track_id',xcol,ycol]].groupby('track_id')
        dfgm = dfg.mean()
        dfgm.dropna(subset=[xcol,ycol],inplace=True)
        x = dfgm[xcol]
        y = dfgm[ycol]
        reg = LinearRegression().fit(x.values.reshape(-1,1),y.values)
        y_pred = reg.predict(x.values.reshape(-1,1))
        r2 = reg.score(x.values.reshape(-1,1),y.values)
        curr_ax.plot(x,y_pred,'k--',label=f'all, r2={r2:.2f}')

        curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5))