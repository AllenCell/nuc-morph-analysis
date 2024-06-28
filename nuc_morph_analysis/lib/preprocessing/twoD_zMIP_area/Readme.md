# twoD_zMIP_area code 
1. `segment_colonies.py` generates 2d segmentations of the brightield colonies and dataframes that save the 2d_zMIP_area of the colony (`colony_area`)
    - each colony takes about 10 minutes to segment (570 frames)
2. `evaluate_method_on_multiple_frames.py` generates figures that show the intermediate steps of the segmentation workflow and final results for multiple images from the dataset. It does this by calling `evaluate_method.py`
3. `make_overlay_image_sequences.py` and `make_overlay_movies.py` generate segmentation overlays on the brighfield images and movies of these overlays respectively for visualization. 

- `twoD_zMIP_helper.py` contains functions that support the above scripts
- `run_colony_segmentation_workflow.py` to run all

# twoD_zMIP_area features
this directory contains code to compute the following:
1. `colony_area`: the 2d zMIP area of the colony (from brightfield FOV)
    - *whole colony feature*
    - segments the brightield image and sums the pixels from the segmentation
    - calculated in `segment_colonies.py`
    
2. `seg_twoD_zMIP_area`: the sum of the 2d zMIP area of all nuclei (from segmentation FOV)
    - *whole colony feature*
    - binarizes the max projection of the segmented FOV and sums pixels
    - note: this is may be slightly different than taking the sum of all values in #4 below because some nuclei may overlap in Z. 
    - calculated in `compute_nuclear_twoD_zMIP_area.py`

3. `nucleus_colony_area_ratio`: the ratio of the sum of the 2d area of all nuclei (#2, `seg_twoD_zMIP_area`) divided by the 2d area of the colony (#1, `colony_area`)
   - *whole colony feature*
   - this represents the crowdednes of the colony in 2d.  could also be named `global_area_crowdedness`