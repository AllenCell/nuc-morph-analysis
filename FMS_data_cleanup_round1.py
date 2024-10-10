#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
verbose=False

# load the parquet file from Sean
df_pq = pd.read_parquet('//allen/aics/software/seanm/NucMorph_TIFF_files_all.parquet')
print('number of files : ', df_pq.shape[0])
print('total file size (TB) : ', np.sum(df_pq['size_TB']))

# add annotations to the dataframe
list_of_dictionaries = df_pq['_annotations'].tolist()
df_ann = pd.DataFrame(list_of_dictionaries)
df = pd.merge(df_pq, df_ann, left_index=True, right_index=True)
#%% define plotting function
def plot_counts_and_size(dfin,bins=np.arange(0,10.5,0.5),titlestr='all files',yscale='log'):
    fig,ax = plt.subplots(1,2,figsize=(6,3),layout='constrained')
    plt.sca(ax[0])
    histoutput = np.histogram(dfin['size_GB'], bins=bins)
    counts = histoutput[0]
    plt.bar(bins[:-1], counts, width=bins[1]-bins[0])
    plt.yscale('log')
    plt.xlabel('Size (GB)')
    plt.ylabel('# of files per bin')
    plt.title(f'COUNT per bin\nTotal number of files : {np.sum(counts)}')

    plt.sca(ax[1])
    #convert count to GB
    tbs_per_bin = dfin.groupby(pd.cut(dfin['size_GB'], bins=bins))['size_TB'].sum()
    plt.bar(bins[:-1], tbs_per_bin, width=bins[1]-bins[0])
    plt.xlabel('Size (GB)')
    plt.ylabel('summed file size per bin (TB)')
    # plt.title('all files\nSIZE per bin\total size : ' + str(np.sum(tbs_per_bin)) + ' TB')
    plt.title(f"SIZE per bin\ntotal size : {np.sum(tbs_per_bin):.2f} TB")
    fig.suptitle(titlestr)
    plt.show()

#%% plot all files and plot small files
plot_counts_and_size(df,bins=np.arange(0,10.5,0.5),titlestr='all files')
df_to_delete = df_pq[df_pq['size_GB'] < 0.5]
plot_counts_and_size(df_to_delete,bins=np.arange(0,0.6,0.01),titlestr='files to delete',yscale='linear')

#%% look at example file names to confirm that these are single cell crops
print('example file names to delete')
[print(f'   {x}') for x in df_to_delete.iloc[0:5]['_name'].tolist()]

print('')
print('number of files < 0.5 GB to delete: ', df_to_delete.shape[0])
print('total size of files < 0.5 GB to delete (TB): ', np.sum(df_to_delete['size_TB']))
print("number of remaining files: ", df.shape[0]-df_to_delete.shape[0])


#%% now save a new parquet to send back to Sean
import os
save_dir = '//allen/aics/assay-dev/users/Frick/forOthersTemp/forSean'
os.makedirs(save_dir, exist_ok=True)
save_name = 'NucMorph_TIFF_files_all_small_TO_DELETE.parquet'
save_path = f'{save_dir}/{save_name}'
df_to_delete.to_parquet(save_path)
print(save_path)