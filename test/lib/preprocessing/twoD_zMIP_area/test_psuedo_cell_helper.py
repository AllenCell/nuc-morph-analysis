import numpy as np
import pandas as pd
import bioio
from bioio import BioImage

from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper
from nuc_morph_analysis.lib.preprocessing import load_data

from nuc_morph_analysis.lib.preprocessing.add_times import get_nearest_frame

NUCLEUS_SIZE=20
DISTANCE_BETWEEN_CELLS=10
CELLS_PER_ROW=10
def add_nucleus_to_image(image, x:int,y:int,z:int, cell_index:int, nucleus_size:int, margin:int)->None:
    base_x = x * (nucleus_size + margin) + margin
    base_y = y * (nucleus_size + margin) + margin
    for x_offset in range(nucleus_size):
        for y_offset in range(nucleus_size):
            current_x = base_x + x_offset
            current_y = base_y + y_offset
            image[z,current_x, current_y] = cell_index



# 
# Method to make a dummy image with 'cells' dropped into a field of zeros.
# drops the cells evenly in a square grid in the field, seperated by margin, with dimensions nucleus_size X nucleus_size
def make_nucleus_image_array(nucleus_size:int=NUCLEUS_SIZE, margin:int=DISTANCE_BETWEEN_CELLS, cells_per_row:int=CELLS_PER_ROW):
    image_square = cells_per_row * (nucleus_size + margin) + margin
    image = np.zeros((1,image_square,image_square),dtype='uint16')
    # 100 cells
    cell_index = 1
    for row in range(cells_per_row):
        for col in range(cells_per_row):

            add_nucleus_to_image(image=image, x=col,y=row,z=0, cell_index=cell_index, nucleus_size=nucleus_size, margin=margin)
            cell_index+=1
    return image

def test_get_pseudo_cell_boundaries():
    # Create a sample dataframe
    raw_data = make_nucleus_image_array()
    
    
    # not available in the installed bioio
    #report = bioio.plugin_feasibility_report(image=raw_data)
    image = BioImage(image=raw_data)
    TIMEPOINT = 0  # not really sure what this is
    colony = 'test'
    RESOLUTION_LEVEL = 0
    reader = image
    
    try:
        df_2d = pseudo_cell_helper.get_pseudo_cell_boundaries(make_nucleus_image_array())
        assert(df_2d['2d_area_nucleus'].values[0] == NUCLEUS_SIZE**2)
        assert(df_2d['2d_area_pseudo_cell'].values[0] == (NUCLEUS_SIZE + DISTANCE_BETWEEN_CELLS)**2)
        assert(len(df_2d) == (CELLS_PER_ROW-2)**2)
        assert(df_2d['2d_area_nuc_cell_ratio'].values[0] == (NUCLEUS_SIZE**2)/((NUCLEUS_SIZE + DISTANCE_BETWEEN_CELLS)**2))

    except Exception as e:
        print(f'Exception in testing get_pseudo_cell_boundaries: {e}')
        assert(False)
    print('hello')
    