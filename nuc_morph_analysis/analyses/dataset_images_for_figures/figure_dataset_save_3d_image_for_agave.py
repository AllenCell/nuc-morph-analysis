# %%
import numpy as np
from aicsimageio.writers import ome_tiff_writer

from nuc_morph_analysis.analyses.dataset_images_for_figures.figure_helper import (
    get_save_dir_and_fig_panel_str,
    return_glasbey_on_dark,
)
from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_info_by_name,
    get_seg_fov_for_dataset_at_frame,
)
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.visualization.example_tracks import EXAMPLE_TRACKS
from nuc_morph_analysis.lib.preprocessing.system_info import PIXEL_SIZE_YX_100x, PIXEL_SIZE_Z_100x
from pathlib import Path

# %%
# create the saving directory if it doesn't exist
figure = "dataset"
panel = "image_for_agave"
savedir, fig_panel_str = get_save_dir_and_fig_panel_str(figure, panel)

# define dataset from which to collect images
# collect information for the dataset
name = "medium"
dataset = get_dataset_info_by_name(name)
timepoint_frame = 48  # choose the timepoints to load (30 minutes) from medium
# define which track_id to box and crop
track_id = EXAMPLE_TRACKS["figure_dataset_processing_workflow"]

# define how many frames to use to create the tracking tails
cell_color = np.asarray([255, 255, 0])

# show segmentations as sum projections of contours to emphasize 3D ness
use_cv_contours_for_3d = False

df = load_dataset_with_features(name)

# %%
# find the label number for the chosen track
row = df[(df.track_id == track_id) & (df.index_sequence == timepoint_frame)]
cell_label_num = row["label_img"].iloc[0]
# very important for color consistency define max value for the colormap based on the max value of the label_img for the whole FOV.
N = int(df.set_index("index_sequence").loc[timepoint_frame, "label_img"].max())

# collect the segmented image at the timepoint of interest
seg_3d_image0 = get_seg_fov_for_dataset_at_frame(name, timepoint_frame)

# now define the ROIs.
full_roi = np.asarray(
    [0, seg_3d_image0.shape[0], 0, seg_3d_image0.shape[1], 0, seg_3d_image0.shape[2]]
)
# %%
from aicsimageio.types import PhysicalPixelSizes

physical_pixel_sizes = PhysicalPixelSizes(PIXEL_SIZE_Z_100x, PIXEL_SIZE_YX_100x, PIXEL_SIZE_YX_100x)
# now define the ROI for the cell of interest by retrieving its crop ROI from the dataframe
crop_widths = [100, 100]  # widths in Y and X; crop 100 pixels around the centroid
cx, cy = row["centroid_x"].iloc[0], row["centroid_y"].iloc[0]
crop_roi = np.asarray(
    [
        0,
        seg_3d_image0.shape[0],
        cy - crop_widths[0],
        cy + crop_widths[0],
        cx - crop_widths[1],
        cx + crop_widths[1],
    ]
)

# the large block below will crop the raw and seg images, create a slice or projection view, and then draw contours and tails and save out the images and figures
# now put these into a list to iterate through
crop_list = [(full_roi, "fullFOV"), (crop_roi, "crop")]
for cropping_roi_seg, extrastr in crop_list:  # iterate through the list

    # define an index_exp for easy cropping
    crop_exp_seg = np.index_exp[
        cropping_roi_seg[0] : cropping_roi_seg[1],
        cropping_roi_seg[2] : cropping_roi_seg[3],
        cropping_roi_seg[4] : cropping_roi_seg[5],
    ]
    seg_3d_image = seg_3d_image0[crop_exp_seg]

    if extrastr == "crop":
        seg_3d_image[seg_3d_image != cell_label_num] = 0

    rgb_array0_255, _, _ = return_glasbey_on_dark(
        N=N, cell_id=cell_label_num, cell_color=cell_color
    )

    ome_tiff_writer.OmeTiffWriter.save(
        seg_3d_image,
        Path(savedir) / f"seg_3d_image_{extrastr}.ome.tif",
        dim_order="ZYX",
        physical_pixel_sizes=physical_pixel_sizes,
    )
    with open(Path(savedir) / "modified_glasbey_colormap.npy", "wb") as f:
        np.save(f, rgb_array0_255)
