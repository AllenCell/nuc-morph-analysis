# %%
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from nuc_morph_analysis.lib.preprocessing import all_datasets
from nuc_morph_analysis.analyses.segmentation_model_validation.seg_model_validation_helper import (
    get_dictonary_of_images_from_manifest,
    align_lr_images_to_hr_images,
    pad_images_in_dictionary_to_same_size,
    resize_low_res_images_in_dict,
    apply_intensity_rescaling_to_raw_images_in_dict,
    determine_label_values_at_zyx_in_all_seg_images,
    crop_all_images_in_dictionary,
    process_and_save_images,
    keep_label_pixels_only_for_seg_images_in_dict,
    add_paths_to_manifest,
    define_FOVid,
)


def save_out_specified_image_pairs_with_overlays(
    fov_list=["IMG_0395.tif", "IMG_0400.tif"], cell_list=["f62af69b9e", "0be9863069"]
):
    """
    This function will save out the specified image pairs with overlays for the specified FOVs and cells.

    Parameters
    ----------
    fov_list : list
        List of FOVids to process.
    cell_list : list
        List of CellIds to process. Only valid if a single FOV is specified in fov_list
    """
    csv_url = all_datasets.segmentation_model_validation_URLs["single_cell_features"]
    df0 = pd.read_csv(csv_url)
    df0 = add_paths_to_manifest(df0)
    df0 = define_FOVid(df0)
    dfmc = df0.groupby("id").agg("first").reset_index()

    # define the figure save directory
    save_dir = Path(__file__).parent / "figures" / "image_pairs"
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)

    if "all" in fov_list:
        pass  # run through all FOVs
    elif len(fov_list) > 0:
        # truncate to the specified FOVs
        dfmc = dfmc[dfmc["id"].isin(fov_list)]

    # now go through each FOV in dfmc
    for mi, (FOVid, dfmcg) in enumerate(dfmc.groupby(["id"])):

        # get the FOV images
        row = dfmcg.iloc[0]
        FOVid = row["id"]

        # collect the single cell manifest for the FOV, which gives details with the gt_seg image as reference
        dfgf = df0.loc[df0["id"] == FOVid, :]

        img_dict = get_dictonary_of_images_from_manifest(row)
        img_dict = resize_low_res_images_in_dict(img_dict)
        # make sure all images are same size
        img_dict = pad_images_in_dictionary_to_same_size(img_dict)
        # perform an alignment to correct for slight offsets
        aligned_dict = align_lr_images_to_hr_images(img_dict)
        # now adjust contrast in all images for viewing in figure
        aligned_dict_rs = apply_intensity_rescaling_to_raw_images_in_dict(aligned_dict)

        ########################################
        # now save out all the images at FOV level
        figdir = save_dir / "FOVs"
        if not figdir.exists():
            figdir.mkdir(exist_ok=True, parents=True)

        # process images into slices and save
        FOVout = process_and_save_images(aligned_dict_rs.copy(), figdir, FOVid)

        ########################################
        # now iterate through single nuclei crops
        figdir = save_dir / "single_nuclei"
        if not figdir.exists():
            figdir.mkdir(exist_ok=True, parents=True)

        # only iterate through non-edge cells
        # dfgf = dfgf[dfgf['edge_cell']==False]
        # make a new column with just the first 10 characters of the CellId
        # this will be used for filtering using cell_list
        dfgf["CellId10"] = dfgf["CellId"].str[0:10]
        cells_in_fov_list = cell_list[mi : mi + 1]
        if "all" in cells_in_fov_list:
            pass
        elif len(cells_in_fov_list) > 0:
            dfgf = dfgf[dfgf["CellId10"].isin(cells_in_fov_list)]

        for cid, dfgc in dfgf.groupby("CellId"):
            # retrieve the z,y,x centroid coordinates
            zyx = np.array(
                [
                    dfgc["centroid_z"].values[0],
                    dfgc["centroid_y"].values[0],
                    dfgc["centroid_x"].values[0],
                ]
            ).astype("int32")

            # label_gt_seg, label_pr_seg = determine_labels_in_seg_images(zyx,img_dict)
            label_dict = determine_label_values_at_zyx_in_all_seg_images(zyx, img_dict)

            if label_dict["pr_seg"] is None:
                # skip if no matching prediction segmentation is found
                continue

            aligned_dict_rs_for_cropping = keep_label_pixels_only_for_seg_images_in_dict(
                aligned_dict_rs, label_dict
            )
            cropped_dict = crop_all_images_in_dictionary(aligned_dict_rs_for_cropping, zyx)

            CellId = dfgc["CellId"].values[0][0:10]
            CROPout = process_and_save_images(cropped_dict, figdir, FOVid, CellId)


if __name__ == "__main__":
    argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Save out specified image pairs with overlays.")
    parser.add_argument(
        "--fov_list",
        nargs="+",
        type=str,
        help="List of FOVids to process. Separate with spaces.",
        default=["IMG_0395.tif", "IMG_0400.tif"],
    )
    parser.add_argument(
        "--cell_list",
        nargs="+",
        type=str,
        help="List of CellIds to process (first 10 characters). Enter `all` to run all CellIds, or enter just one CellId for the FOV. Separate with spaces.",
        default=["f62af69b9e", "0be9863069"],
    )
    args = parser.parse_args()
    save_out_specified_image_pairs_with_overlays(fov_list=args.fov_list, cell_list=args.cell_list)
