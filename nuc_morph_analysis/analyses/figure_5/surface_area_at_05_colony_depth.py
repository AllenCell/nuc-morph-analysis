# %%
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data 
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
import numpy as np
import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.preprocessing import add_times
#%%
df = load_dataset_with_features('all_baseline',load_local=True)
df_full = filter_data.all_timepoints_full_tracks(df)
#%%
# now validate behavior of digitize_time_column
add_times.validate_dig_time_with_plot()
df_full = add_times.digitize_time_column(df_full,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')

colony_list = ['small','medium','large']
# top row of plots will be ycol = dxdt_48_volume, xcol = index_sequence
# bottom row of plots will be ycol = dxdt_48_volume, xcol = dig_time
# each column will be a different colony
def group_and_extract(dfcc,xcol,ycol):
    grouper = dfcc.groupby(xcol)
    dfg = grouper[ycol].agg([np.nanmean,np.nanstd,'count',lambda x: np.nanpercentile(x,5),lambda x: np.nanpercentile(x,95)])
    dfg.rename(columns={'<lambda_0>':'5th','<lambda_1>':'95th'},inplace=True)
    return dfg

  

#%%
import pandas as pd

def add_feature_at(df,feature_list):
    """
    Adds an feature at a frame of interest to the dataframe.

    Parameters:
    df: DataFrame
        The dataframe
    frame_column: str
        The name of the column that contains the frame at which to calculate the feature.
    feature: str
        The name of the feature to add.
    feature_column: str
        The name of the column that contains the feature.
    multiplier: float, optional
        A multiplier to apply to the feature. Default is 1.

    Returns:
    df: DataFrame
        The dataframe with the added feature column.
    """
    dfn=df.copy()
    dfn = add_times.digitize_time_column(dfn,0,1,step_size=0.02,time_col='normalized_time',new_col='dig_time')
    dfsub = dfn[dfn['dig_time'] ==0.5]
    dft = dfsub[['track_id'] + feature_list].groupby('track_id').mean()
    dft.rename(columns={x : f"{x}_at_05" for x in feature_list},inplace=True)
    dfout= dfn.merge(dft,on='track_id',suffixes=('','_at_05'))
    return dfout



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




#%%
nrows = 2
ncols = len(colony_list)

df_full = add_feature_at(df_full,["volume","height","SA_vol_ratio"])
xcol1 = 'index_sequence'
xcol2 = 'colony_depth'
groupbycol = 'dig_time'
for plot_type in ['mean']:
    for ycol in ['SA_vol_ratio_at_05','volume_at_05','height_at_05','SA_vol_ratio','volume','height']:

        fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*2.5),constrained_layout=True,
                        sharey=True)
        assert type(ax) == np.ndarray # for mypy

        for ci,colony in enumerate(colony_list):
            dfc = df_full[df_full['colony']==colony]

            # top column
            # dfg = group_and_extract(dfc,xcol1,ycol)
            dfg = dfc[['track_id',xcol1,ycol]].groupby('track_id')
            x = dfg[xcol1].mean()
            y = dfg[ycol].mean()
            curr_ax = ax[0,ci]
            curr_ax.scatter(x,y,s=1)
            curr_ax.set_xlabel(xcol1)
            curr_ax.set_ylabel(ycol)

            # curr_ax = plot_dfg(dfg,xcol1,ycol,"",curr_ax,plot_type=plot_type)
            

            # bottom column
            # top column
            dfg = dfc[['track_id',xcol2,ycol]].groupby('track_id')
            x = dfg[xcol2].mean()
            y = dfg[ycol].mean()
            curr_ax = ax[1,ci]
            curr_ax.scatter(x,y,s=1)
            curr_ax.set_xlabel(xcol2)
            curr_ax.set_ylabel(ycol)

        curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5))
    

#%%

df_full = add_feature_at(df_full,["volume","height","SA_vol_ratio"])
xcol1 = 'colony_time'
for plot_type in ['mean']:
    for ycol in ['height','height_at_05','SA_vol_ratio_at_05','SA_vol_ratio','volume','volume_at_05']:

        assert type(ax) == np.ndarray # for mypy
        fig,curr_ax = plt.subplots(1,1,figsize=(3,2))
        for ci,colony in enumerate(colony_list):
            dfc = df_full[df_full['colony']==colony]

            # top column
            # dfg = group_and_extract(dfc,xcol1,ycol)
            dfg = dfc[['track_id',xcol1,ycol]].groupby('track_id')
            x = dfg[xcol1].mean()
            y = dfg[ycol].mean()
            assert type(curr_ax) == plt.Axes
            curr_ax.scatter(x,y,s=1)
            curr_ax.set_xlabel(xcol1)
            curr_ax.set_ylabel(ycol)

        curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5))

#%%
from sklearn.linear_model import LinearRegression
df_full = add_feature_at(df_full,["volume","height","SA_vol_ratio"])
xcol1 = 'normalized_colony_depth'
for plot_type in ['mean']:
    for ycol in ['height','height_at_05','SA_vol_ratio_at_05','SA_vol_ratio','volume','volume_at_05']:

        assert type(ax) == np.ndarray # for mypy
        fig,curr_ax = plt.subplots(1,1,figsize=(3,2))
        for ci,colony in enumerate(colony_list):
            dfc = df_full[df_full['colony']==colony]
            dfc = dfc[(dfc['colony_time']>420) & (dfc['colony_time']<440)]

            # top column
            # dfg = group_and_extract(dfc,xcol1,ycol)
            dfg = dfc[['track_id',xcol1,ycol]].groupby('track_id')
            x = dfg[xcol1].mean()
            y = dfg[ycol].mean()

            reg = LinearRegression().fit(x.values.reshape(-1,1),y.values)
            y_pred = reg.predict(x.values.reshape(-1,1))
            r2 = reg.score(x.values.reshape(-1,1),y.values)
            curr_ax.plot(x,y_pred,linestyle='--',label=f'{colony}, r2={r2:.2f}')
            assert type(curr_ax) == plt.Axes
            curr_ax.scatter(x,y,s=1)
            curr_ax.set_xlabel(xcol1)
            curr_ax.set_ylabel(ycol)

        # now all
        dfc = df_full
        dfc = dfc[(dfc['colony_time']>420) & (dfc['colony_time']<440)]
        dfg = dfc[['track_id',xcol1,ycol]].groupby('track_id')
        x = dfg[xcol1].mean()
        y = dfg[ycol].mean()
        reg = LinearRegression().fit(x.values.reshape(-1,1),y.values)
        y_pred = reg.predict(x.values.reshape(-1,1))
        r2 = reg.score(x.values.reshape(-1,1),y.values)
        curr_ax.plot(x,y_pred,'k--',label=f'all, r2={r2:.2f}')


        curr_ax.legend(loc='center left',bbox_to_anchor=(1.05,0.5))
        curr_ax.set_title(f"time_between_420_and_440")