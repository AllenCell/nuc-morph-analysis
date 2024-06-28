# %%
from pathlib import Path
from aicsimageio.writers import two_d_writer
from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_original_file_reader,
    get_channel_index,
)
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import twoD_zMIP_helper
import matplotlib.pyplot as plt
import numpy as np


def evaluate_method(dataset="small", time=10):
    reader = get_dataset_original_file_reader(dataset)
    C = get_channel_index(dataset, "bright")
    fov_bri_data = reader.get_image_dask_data("ZYX", C=C, T=time).compute()
    feats = {
        "dataset": dataset,
        "index_sequence": time,
    }

    # now segment the colony
    feats, segmentation_intermediates = twoD_zMIP_helper.segment_colony_brightfield(
        fov_bri_data, collect_intermediates=True
    )
    feats2, brightfield_img_intermediates = (
        twoD_zMIP_helper.get_best_contrast_brightfield_img_slices(
            fov_bri_data, collect_intermediates=True
        )
    )

    # now plot all the intermediates of segmentation on a figure with subplots
    panels = {}
    panels.update({"best_contrast_img": feats2["best_contrast_img"]})
    panels.update(segmentation_intermediates)
    panels.update(
        {
            "drawn_contour_overlay": twoD_zMIP_helper.draw_contours_on_infocus_slice(
                panels["best_contrast_img"], panels["hole_fill2=final"]
            )
        }
    )
    nrows = 3
    ncols = np.ceil(len(panels) / nrows).astype(int)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))
    ax = ax.flatten()
    for i, (k, v) in enumerate(panels.items()):
        ax[i].imshow(
            v,
            cmap="gray" if v.dtype == "uint8" else "viridis",
            interpolation="nearest",
        )
        ax[i].set_title(k)
    for i in range(len(ax)):
        ax[i].axis("off")

    # save in the local_storage/figures directory
    save_dir = Path(__file__).parent / "figures" / "evaluation"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # save the figure
    fig.suptitle(f"Segmentation Intermediates of Brightfield Image from {dataset} at frame={time}")
    fig.savefig(save_dir / f"{dataset}_seg_intermediates_t{str(time).zfill(3)}.png")
    plt.show()

    # now plot all the stdev profile of brightfield on a figure
    nrows = 1
    ncols = 3
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4 * nrows),
    )
    fig.subplots_adjust(wspace=0.7)
    ax[0].imshow(
        brightfield_img_intermediates["in_focus_img"],
        cmap="gray",
        interpolation="nearest",
        vmin=np.percentile(brightfield_img_intermediates["in_focus_img"], 5),
        vmax=np.percentile(brightfield_img_intermediates["in_focus_img"], 95),
    )
    ax[0].set_title("In Focus Slice")
    ax[1].imshow(
        brightfield_img_intermediates["best_contrast_img"],
        cmap="gray",
        interpolation="nearest",
        vmin=np.percentile(brightfield_img_intermediates["best_contrast_img"], 5),
        vmax=np.percentile(brightfield_img_intermediates["best_contrast_img"], 95),
    )
    ax[1].set_title("Best Contrast Slice")

    y = brightfield_img_intermediates["profile"]
    x = list(range(len(y)))

    ax[2].plot(x, y, "k-", label="Standard Deviation")
    for ki, key in enumerate([x for x in brightfield_img_intermediates.keys() if "slice" in x]):
        print(key)
        if brightfield_img_intermediates[key] is not np.nan:
            ax[2].plot(
                x[brightfield_img_intermediates[key]],
                y[brightfield_img_intermediates[key]],
                "o",
                markersize=6,
                label=key,
            )

    xx = (
        x[brightfield_img_intermediates["first_peak_slice"]],
        x[brightfield_img_intermediates["second_peak_slice"]],
    )
    yy = (
        y[brightfield_img_intermediates["first_peak_slice"]],
        y[brightfield_img_intermediates["second_peak_slice"]],
    )
    yy = np.ones_like(yy) * np.max(yy) * 1.1
    ax[2].plot(xx, yy, "b--", label="frames used for segmentation", linewidth=2)
    ax[2].fill_between(xx, y1=np.min(y) / 1.1, y2=yy, alpha=0.1, color="blue")
    ax[2].set_ylabel("Standard Deviation/Mean at a given Z")
    ax[2].set_xlabel("Z slice index")
    ax[2].set_title(
        f"Standard Deviation/Mean Profile of Brightfield Image from \n {dataset} at frame={time}"
    )
    ax[2].legend(loc="center left", bbox_to_anchor=(1.1, 0.5))

    # save in the local_storage/figures directory
    save_dir = Path(__file__).parent / "figures" / "evaluation"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # save the figure
    fig.savefig(save_dir / f"{dataset}_slice_choice_intermediates_t{str(time).zfill(3)}.png")
    plt.show()

    # now save out the overlay image on its own
    save_dir = Path(__file__).parent / "figures" / "evaluation" / "overlay_images"
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    two_d_writer.TwoDWriter.save(
        panels["drawn_contour_overlay"], save_dir / f"{dataset}_overlay_t{str(time).zfill(3)}.png"
    )


if __name__ == "__main__":
    evaluate_method()
