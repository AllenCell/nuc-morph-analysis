#%%
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from lxml import etree
from aicspylibczi import CziFile
import skimage.exposure as skex
import skimage.io as skio
from pathlib import Path
from multiprocessing import Pool, cpu_count

NWORKERS = np.min([32,cpu_count()])
LOW_PIX_INT = 15000
HIGH_PIX_INT = 32000
CHANNEL_IDX = 0
SAVE_DIR = Path(__file__).parents[6] / 'data' / 'TILEDexports'
print(SAVE_DIR)


def collect_metadata(reader):
    meta0 = reader.metadata    
    #convert metadata to lxml
    metastr = ET.tostring(meta0).decode("utf-8")
    meta = etree.fromstring(metastr)

    feats={}
    feats['file'] = file_path
    feats['parent_file'] = file_path

    regions = list(meta.findall('.//ParameterCollection/ImageFrame'))
    txtout = regions[0].text
    frame_size_pixels = eval(txtout)

    #number of pixels in each dimension
    feats['shape'] = tuple((frame_size_pixels[-2],frame_size_pixels[-1])) #number of pixels in each dimension

    ImagePixelDistancesList = meta.findall('.//ParameterCollection/ImagePixelDistances')
    for ip in ImagePixelDistancesList[0:1]: #only choose the first camera
        feats['ImagePixelDistances'] = tuple(eval(ip.text))
        feats['totalmagnification'] = eval(ip.getparent().find('./TotalMagnification').text)

    xypxsize = (np.asarray(feats['ImagePixelDistances'])/feats['totalmagnification'])
    feats['pixelSizes'] = (xypxsize[0],xypxsize[1])#units of um
    feats['imgsize_um'] = tuple([x*y for x,y in zip(feats['pixelSizes'] ,feats['shape'])])
    active_tile_region = [xx.getparent() for xx in meta.findall('.//TileRegion/IsUsedForAcquisition') if xx.text=='true'][0]
    feats['ContourSize_um'] = eval(active_tile_region.findall('./ContourSize')[0].text)
    feats['Rows'] = eval(active_tile_region.findall('./Rows')[0].text)
    feats['Columns'] = eval(active_tile_region.findall('./Columns')[0].text)

    Scenes = meta.findall('.//SizeS')[0].text
    feats['numberOfScenes'] = int(Scenes)

    Channels = meta.findall('.//SizeC')[0].text
    feats['numberOfChannels'] = int(Channels)

    Time = meta.findall('.//SizeT')[0].text
    feats['numberOfTimePoints'] = int(Time)
    return feats

def compute_overlaps(feats):
    """
    compute overlaps in pixels between tiles

    Parameters
    ----------
    feats : dict
        dictionary of metadata

    Returns
    -------
    overlaplist : list
        list of overlaps in pixels between tiles
    px_w_stich_list : list
        list of pixel sizes for each tile 
    """
    overlaplist=[]
    px_w_stich_list=[]
    for row_col, i in zip(['Rows','Columns'],range(2)):
        px_no_stich = np.round(feats['imgsize_um'][i]*feats[row_col]/feats['pixelSizes'][0],0)
        px_w_stich = np.round(feats['ContourSize_um'][i]/feats['pixelSizes'][0],0)
        px_no_stich,px_w_stich
        overlap = px_no_stich-px_w_stich
        overlaplist.append(np.uint16(overlap))
        print(row_col,px_no_stich,px_w_stich,overlap)
        px_w_stich_list.append(np.uint16(px_w_stich))
    return overlaplist,px_w_stich_list

def auto_determine_tile_xy_locations(file_path,num_tiles):
    czi = CziFile(file_path) 
    dfxylist = []    
    for m in range(num_tiles):
        o= czi.read_subblock_metadata(M=m,
                                        C=0,
                                        T=0,
                                        Z=0,
                                        S=scene)
        submetastr = o[0][1]
        submeta = ET.fromstring(submetastr)
        x = np.float32(submeta.find('.//StageXPosition').text)
        y = np.float32(submeta.find('.//StageYPosition').text)
        # print(m,x,y)
        feats={
            'm':m,
            'x':x,
            'y':y,}
        dfxylist.append(pd.DataFrame(
            data=feats.values(),
            index=feats.keys(),
                            ).T,
                        )

    dfxy = pd.concat(dfxylist)
    dfxy['xmin']=dfxy['x']-dfxy['x'].min()
    dfxy['ymin']=dfxy['y']-dfxy['y'].min()
    dfxy['xt'] = np.round(dfxy['xmin']/dfxy['xmin'].max()*(len(np.unique(dfxy['xmin']))-1),0).astype('uint8')
    dfxy['yt'] = np.round(dfxy['ymin']/dfxy['ymin'].max()*(len(np.unique(dfxy['ymin']))-1),0).astype('uint8')
    return dfxy

def basic_stack_and_project_tiles(dfxy,px_w_stich_list,imgstack,overlaplist):
    stack_list=[]
    for m in range(dfxy.shape[0]):
        imgout = np.zeros([px_w_stich_list[1],px_w_stich_list[0]])*np.nan
        img_sub = imgstack[m]
        xt = dfxy.set_index('m').loc[m,'xt']
        yt = dfxy.set_index('m').loc[m,'yt']
        # print(xt,yt)

        y1 = (img_sub.shape[0])*yt - ((overlaplist[1]*yt)//2)
        y2 = (((img_sub.shape[0])*(yt+1)) - ((overlaplist[1]*yt)//2))-1
        y2 = (((img_sub.shape[0])*(yt+1)) - (np.ceil((overlaplist[1]*yt)//2)).astype('uint16'))

        x1 = (img_sub.shape[1]*xt) - (np.ceil((overlaplist[0]*xt)/2)).astype('uint16')
        x2 = ((img_sub.shape[1])*(xt+1) - (np.floor((overlaplist[0]*xt)/2)).astype('uint16'))
        # print(y2-y1)
        # print(x2-x1)
        the_crop = np.index_exp[
            y1 :  y2,
            x1  : x2,
            # (img_sub.shape[1]-1)*xt  :  (((img_sub.shape[1])*(xt+1)) - (overlaplist[0]*xt)),
        ]

        # print(the_crop)


        # print(img2.shape,the_crop2)
        imgout[the_crop] = img_sub
        stack_list.append(imgout)
    
    #max project the stack (the stack becomes M x Y x X)
    stiched_img0 = np.nanmax(np.stack(stack_list),axis=0)
    return stiched_img0
    
def def_save_path(file_path,scene,tval,SAVE_DIR):
    sep = os.sep
    extra_dir = f"={Path(file_path).stem}-export{sep}S{str(scene+1).zfill(2)}"
    savedir = SAVE_DIR / extra_dir

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savename = f"{Path(file_path).stem}-S{str(scene+1).zfill(2)}-T{str(tval+1).zfill(4)}.tif"
    savepath = f"{savedir}{sep}{savename}"
    return savepath

def process_timepoint_input(args):
    return process_timepoint(*args)

def process_timepoint(reader,scene,tval,CHANNEL_IDX,SAVE_DIR):
    delayed_chunk = reader.get_image_dask_data("MYX", T=tval, C=CHANNEL_IDX)
    imgstack = delayed_chunk.compute()
    dfxy = auto_determine_tile_xy_locations(file_path,imgstack.shape[0])
    #stack the images and then max projet the stack to stich
    stiched_img0 = basic_stack_and_project_tiles(dfxy,px_w_stich_list,imgstack,overlaplist)
    
    # downsample 2x2
    # stiched_img0 = stiched_img0[::2,::2]
    
    stiched_img = skex.rescale_intensity(stiched_img0,
                                in_range=(LOW_PIX_INT,
                                        HIGH_PIX_INT),
                                out_range='uint8').astype('uint8')
    
    savepath = def_save_path(file_path,scene,tval,SAVE_DIR)

    skio.imsave(savepath,
                np.uint8(stiched_img),
                check_contrast=False)

    return savepath

def export_all_timepoints_for_scene(reader,scene,CHANNEL_IDX,SAVE_DIR,parrallelize=True):
    print("scene=",scene)
    reader.set_scene(scene)
    T = reader.dims['T']
    TList = np.arange(0,T[0],1).astype('uint16')
    # for ti in tqdm(range(len(TList))):
    #     tval = TList[ti]
    #     #read image chunk
    args_list = [(reader,scene,tval,CHANNEL_IDX,SAVE_DIR) for tval in TList]
    if parrallelize:
        with Pool(NWORKERS) as p:

            # results = list(tqdm(p.imap(lambda tval: process_timepoint(reader,scene,tval,CHANNEL_IDX,SAVE_DIR),TList),total=len(TList)))
            results = list(tqdm(p.imap(process_timepoint_input,args_list),total=len(TList)))
    else:
        results = []
        results.append([process_timepoint(*args) for args in args_list])
    return results


        
#%%
from bioio import BioImage
from tqdm import tqdm

# find file and determine size
file_path = '/allen/aics/assay-dev/MicroscopyData/Leveille/2023/20230720/2023-07-20/AD00004745_20230720_AICS13_L01-01.czi/AD00004745_20230720_AICS13_L01-01_AcquisitionBlock2.czi/AD00004745_20230720_AICS13_L01-01_AcquisitionBlock2_pt2.czi'
reader = BioImage(file_path,reconstruct_mosaic=False)
print(reader.dims) # <Dimensions [M: 9, T: 576, C: 1, Z: 1, Y: 1248, X: 1848]>

# determine number of scenes
print(reader.scenes)
#%%
feats = collect_metadata(reader)
overlaplist,px_w_stich_list = compute_overlaps(feats)
scene_list = reader.scenes
for scenetuple in enumerate(scene_list):
    scene=scenetuple[0]
    print(scene)
    results = export_all_timepoints_for_scene(reader,scene,CHANNEL_IDX,SAVE_DIR)
    # print first file path
    print(results[0])
