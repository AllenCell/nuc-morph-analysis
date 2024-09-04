import numpy as np

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




