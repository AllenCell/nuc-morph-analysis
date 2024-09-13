#%%
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import watershed_workflow
from nuc_morph_analysis.lib.preprocessing import load_data

colony = 'medium'
timepoint = 48

reader = load_data.get_dataset_segmentation_file_reader(colony)

df_2d_list=[]
for resolution_level in [0,1]:
    df_2d = watershed_workflow.get_image_and_run(colony, timepoint, reader, resolution_level, return_img_dict=False)
    df_2d_list.append(df_2d)

#%%
import pandas as pd
import matplotlib.pyplot as plt
# examine correlation between nucleus_area for all dataframes
df1 = df_2d_list[0]
df2 = df_2d_list[1]
dfm = pd.merge(df1, df2, on='label_img', suffixes=('_0', '_1'))

# now plot correlation
feature_list = ['2d_area_nucleus','2d_area_pseudo_cell','2d_area_cyto','inv_cyto_density','2d_area_nuc_cell_ratio']
for feature in feature_list:
    fig,ax = plt.subplots()
    ax.scatter(dfm[feature+'_0'], dfm[feature+'_1'])
    ax.set_xlabel(feature+'_0')
    ax.set_ylabel(feature+'_1')
    ax.set_title(feature)
    plt.show()

