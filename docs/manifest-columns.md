The main manifest is an Apache Parquet file. You can open it with `pandas` (must also have the python package `pyarrow` installed).
```python
>>> import pandas as pd
>>> main_df = pd.read_parquet("path/to/main_manifest.parquet")
```

Column Name | Data type | Units | Description | Computed by
-|-|-|-|-
CellId | string | | Unique identifier for the row. One row represents one nucleus at one time. | cyto-dl
index_sequence | int | frames | Number of frames since the start of the movie. First frame is 0. Frames are each 5 minutes apart. |
track_id | string | | Unique identifier for one nucleus over time. | aics-timelapse-tracking and nuc-morph-analysis
dataset | string | | The name of the colony. | nuc-morph-analysis
volume | int | voxels | Number of voxels in the nucleus segmentation | morflowgenesis
height | int | voxels | Distance from bottom of nucleus segmentation to top | morflowgenesis
scale_micron | float[] | micron | Voxel size in microns for [Z, Y, X] dimensions |
centroid_x | int | voxels | Horizontal x position of the nucleus segmentation's centroid relative to the field of view. | morflowgenesis
centroid_y | int | voxels | Horizontal y position of the nucleus segmentation's centroid relative to the field of view. | morflowgenesis
centroid_z | int | voxels | Vertical position of the nucleus segmentation's centroid relative to the field of view. The bottom of the FOV is 0. | morflowgenesis
predicted_formation | string | frames | |
predicted_breakdown | string | frames | |
label_img | int | | | cyto-dl
is_tp_outlier | bool | | | 
roi | | | | cyto-dl
fov_edge | bool | | 1 if the nucleus is on the edge of the FOV, 0 otherwise | cyto-dl
parent_id | int | Track ID | If a parent cell was manually identified, this is the track_id of the parent. If the cell was manually identified to have no parent, this is -1. Otherwise NaN. | nuc-morph-analysis
termination | int | NaN, 0, 1, or 2 | 0 - track terminates by dividing. 1 - track terminates by going off the edge of the FOV. 2 - track terminates by apoptosis. | nuc-morph-analysis
neighbors | string | | String representation of a list of CellIds. These are the adjacent nuclei, based on only nuclei centroids. | nuc-morph-analysis
neigh_distance | float | voxels | Mean distance from this nucleus's centroid to the centroids of neighboring nuclei. | nuc-morph-analysis
density  | float | 1 / voxels^2 | 1 / neigh_distance^2 | nuc-morph-analysis
colony_depth | int | nuclei | Number of nuclei from the outermost nuclei in the field of view. The outermost ring of nuclei have colony depth 1. This column is not particularly meaningful after the colony outgrows the field of view. | nuc-morph-analysis
NUC_shcoeffs_L?M?S | float | | | morflowgenesis
NUC_shcoeffs_L?M?C | float | | | morflowgenesis
NUC_PC? | float | | | nuc-morph-analysis
