from pathlib import Path
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from aicsimageio.writers import two_d_writer
from aicsimageio import AICSImage
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area.twoD_zMIP_helper import (
    draw_contours_on_infocus_slice,
)


def run_make_overlays_in_parallel(args):
    time, df, dataset, save_dir = args
    run_make_overlays(time, df, dataset, save_dir)


def run_make_overlays(time, df, dataset, save_dir):
    dft = df[df["index_sequence"] == time]

    # load the best_contrast_brightfield_img_slice
    best_contrast_img = AICSImage(dft["best_contrast_img_path"].values[0]).get_image_data("YX")

    # load the segmented_colony_mask
    seg_img = AICSImage(dft["processed_img_path"].values[0]).get_image_data("YX")

    # draw the contours
    contours = draw_contours_on_infocus_slice(best_contrast_img, seg_img)
    two_d_writer.TwoDWriter.save(
        contours, save_dir / f"{dataset}_t{str(time).zfill(3)}.png", dim_orer="YXS"
    )


def run_script(dataset_list):
    num_cores = cpu_count()
    print(f"Number of cores available: {num_cores}")

    img_list = []
    for dataset in dataset_list:
        print(f"Processing dataset: {dataset}")

        # load all csvs in the directory
        csv_dir = Path(__file__).parent / "local_storage" / "colony_seg_outputs" / dataset / "csvs"
        csv_list = list(csv_dir.glob("*.csv"))

        # assemble them into a dataframe
        df = pd.concat([pd.read_csv(csv) for csv in csv_list])
        df.sort_values("index_sequence", inplace=True)

        save_dir = Path(__file__).parent / "local_storage" / "overlay_images" / dataset
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        time_list = df.index_sequence.unique()
        args_list = [(t, df, dataset, save_dir) for t in time_list]
        with Pool(int(num_cores)) as p:

            for _ in tqdm(
                p.imap_unordered(run_make_overlays_in_parallel, args_list), total=len(args_list)
            ):
                pass


if __name__ == "__main__":
    run_script(dataset_list=["small"])
