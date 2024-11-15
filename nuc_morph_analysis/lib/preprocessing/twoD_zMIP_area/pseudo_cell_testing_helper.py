import numpy as np
# functions for testing
NUCLEUS_SIZE=20
DISTANCE_BETWEEN_CELLS=10
CELLS_PER_ROW=10

def _add_nucleus_to_image(image, x:int,y:int,z:int, cell_index:int, nucleus_size:int, margin:int)->None:
    base_x = x * (nucleus_size + margin) + margin
    base_y = y * (nucleus_size + margin) + margin
    for x_offset in range(nucleus_size):
        for y_offset in range(nucleus_size):
            current_x = base_x + x_offset
            current_y = base_y + y_offset
            image[z, current_x, current_y] = cell_index


# Method to make a dummy image with CELLS_PER_ROW x CELLS_PER_ROW cells dropped into a field of zeros.
# drops the cells evenly in a square grid in the field, seperated by margin, with dimensions nucleus_size X nucleus_size
def make_nucleus_image_array(nucleus_size:int = NUCLEUS_SIZE, margin:int = DISTANCE_BETWEEN_CELLS, cells_per_row:int = CELLS_PER_ROW):
    image_square = cells_per_row * (nucleus_size + margin) + margin
    image = np.zeros((1, image_square,image_square), dtype='uint16')
    cell_index = 1  # numeric label of the cell/nucleus
    for row in range(cells_per_row):
        for col in range(cells_per_row):
            _add_nucleus_to_image(image=image, x=col,y=row,z=0, cell_index=cell_index, nucleus_size=nucleus_size, margin=margin)
            cell_index+=1
    return image