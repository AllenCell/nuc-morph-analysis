import numpy as np
import pandas as pd
import bioio
from bioio import BioImage

from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper
from nuc_morph_analysis.lib.preprocessing import load_data

from nuc_morph_analysis.lib.preprocessing.add_times import get_nearest_frame

CELL_SIZE=5
DISTANCE_BETWEEN_CELLS=2
def add_cell_to_image(image, x:int,y:int,z:int, cell_index:int, cell_size:int, margin:int)->None:
    base_x = x * (cell_size + margin) + margin
    base_y = y * (cell_size + margin) + margin
    for x_offset in range(cell_size):
        for y_offset in range(cell_size):
            current_x = base_x + x_offset
            current_y = base_y + y_offset
            image[current_x, current_y, z] = cell_index


# 
# Method to make a dummy image with 'cells' dropped into a field of zeros.
# drops the cells evenly in a square grid in the field, seperated by margin, with dimensions cell_size X cell_size
# default cell_size is 5, margin = 2
def make_cell_image_array(image_square:int=1000, cell_size:int=CELL_SIZE, margin:int=DISTANCE_BETWEEN_CELLS, cells_per_row:int=10):
    image = np.zeros((image_square,image_square,1))
    # 100 cells
    cell_index = 1
    for row in range(cells_per_row):
        for col in range(cells_per_row):
            add_cell_to_image(image=image, x=col,y=row,z=0, cell_index=cell_index, cell_size=cell_size, margin=margin)
            cell_index+=1
    return image

def test_SOMETHING():
    # Create a sample dataframe
    raw_data = make_cell_image_array()
    
    test = False
    
    # not available in the installed bioio
    #report = bioio.plugin_feasibility_report(image=raw_data)
    image = BioImage(image=raw_data)
    TIMEPOINT = 0  # not really sure what this is
    colony = 'test'
    RESOLUTION_LEVEL = 1
    foo = image
    reader = image
    
    if not test:
        # set the details
        TIMEPOINT = 48
        colony = 'medium'
        RESOLUTION_LEVEL = 1
        # load the segmentation image
        reader = load_data.get_dataset_segmentation_file_reader(colony)
        if RESOLUTION_LEVEL>0:
            reader.set_resolution_level(RESOLUTION_LEVEL)
    df_2d, img_dict = pseudo_cell_helper.get_pseudo_cell_boundaries(
        colony=colony, 
        timepoint=TIMEPOINT, 
        reader=reader, 
        resolution_level=RESOLUTION_LEVEL, 
        return_img_dict=True)
    print('hello')
    assert(len(df_2d) == 177 )
    assert(False)

