import numpy as np
import pandas as pd
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path
from multiprocessing import Pool
from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_seg_fov_for_dataset_at_frame,
    load_dataset,
)
import fire


def main(manifest: str, output: str, n: int = 1, process_images: bool = True):
    """
    Convert a nucmorph-formated dataset into the Cell Tracking Challenge format.

    Parameters
    ----------
    manifest: str
        Path to the manifest file.
    output: str
        Path to the output directory. This directory will be created if it
        does not exist.
    n: int
        Number of workers to use for multiprocessing. Default is 1.
    process_images: bool
        Whether or not to process the segmentation images. Default is True.
    """
    converter = CTCConverter(manifest, output, n, process_images)
    converter.run()
    return


class CTCConverter:
    """
    This class is used to convert a nucmorph-formated dataset, images included,
    into the Cell Tracking Challenge format.

    Tracking data is formatted as a .txt file where each line is a single cell
    track. Each line is formatted as follows:

        <track number> <start timepoint> <end timepoint> <parent track number>

    Track numbers are 1-indexed and correspond to unique integer label in the
    fov segmentations and is consistent across timepoints. For nucmorph data
    converted to the CTC format, the track number for a nucleus will be the
    same as its track_id in the original manifest. If a cell has no parent,
    the parent track number is 0.

    As the conversion requires that all segmentation files be relabeled to
    match a cell's track_id, it is best to run this in slurm using a node
    with a large memory and number of cores.

    Relabeled segmentation files will be saved in a subdirectory called 'TRA'
    and a manifest of paths to the transfer function images will be saved in a
    subdirectory called 'GT'.
    """

    def __init__(
        self,
        manifest: str,
        output_dir: str,
        n_workers: int = 1,
        process_images: bool = True,
        lineage_col: str = "None",
    ):
        """
        Initialize the converter.main

        Parameters
        ----------
        manifest: str
            Name or path to the manifest file.
        output_dir: str
            Path to the output directory. This directory will be created if it
            does not exist.
        n_workers: int
            Number of workers to use for multiprocessing. Default is 1.
        process_images: bool
            Whether or not to process the segmentation images. Default is True.
        lineage_col: str
            Column in the manifest file that contains the tracking label that is
            unique to cell lineages. If None, the lineage_id will be the same
            as the track_id. Default is None. (should be either 'lineage_id'
            for old datasets or 'family_id' in new datasets)
        """

        if Path(manifest).exists():
            read_cols = ["index_sequence", "track_id", "colony", "label_img", "in_list", "node_id"]
            if lineage_col != "None":
                read_cols.append(lineage_col)

            self.src_df = pd.read_csv(manifest, usecols=read_cols)

            if lineage_col == "None":
                self.src_df["lineage_id"] = self.src_df["track_id"]
        else:
            self.src_df = load_dataset(manifest)

        if lineage_col != "lineage_id":
            self.src_df["lineage_id"] = self.src_df[lineage_col]
        self.output_dir = Path(output_dir)
        self.n_workers = n_workers
        self.process_images = process_images

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / "TRA").mkdir(exist_ok=True, parents=True)
        (self.output_dir / "GT").mkdir(exist_ok=True, parents=True)
        return

    def run(self):
        """
        Run the conversion.
        """
        self.generate_track_file()

        if self.process_images:
            with Pool(self.n_workers) as p:
                for T, df_T in self.src_df.groupby("index_sequence"):
                    colony = df_T["colony"].unique()[0]

                    p.apply_async(self.relabel_segmentation, args=(T, colony, df_T))
                p.close()
                p.join()
        return

    def generate_track_file(self):
        """
        Generate a track file for the CTC format and save a manifest of the raw
        image files to the GT folder
        """
        with open(self.output_dir / "track.txt", "w") as f:
            for track, df_track in self.src_df.groupby("track_id"):
                start_idx = df_track["index_sequence"].min()
                end_idx = df_track["index_sequence"].max()

                if (
                    df_track.iloc[0]["lineage_id"] == 0
                    or df_track.iloc[0]["lineage_id"] == df_track.iloc[0]["track_id"]
                ):
                    parent = 0
                else:
                    lin_id = df_track.iloc[0]["lineage_id"]
                    df_prev = self.src_df.query(
                        f"lineage_id == {lin_id} and index_sequence < {start_idx}"
                    )

                    parent = 0
                    last_idx = 0
                    for p_id in df_prev["track_id"].unique():
                        df_parent = self.src_df.query(f"track_id == {p_id}")
                        p_end = df_parent["index_sequence"].max()
                        if p_end < start_idx and p_end > last_idx:
                            parent = p_id
                            last_idx = p_end

                f.write(f"{int(track)} {int(start_idx)} {int(end_idx)} {int(parent)}\n")

    def relabel_segmentation(self, T: int, colony: str, df_seg: pd.DataFrame):
        """
        Relabel a segmentation file to match the track_id of each cell.

        Parameters
        ----------
        T: int
            The timepoint of the segmentation file.
        seg_fn: str
            Path to the segmentation file.
        df_seg: pd.DataFrame
            DataFrame containing the segmentation data for the current frame.
        """

        img = get_seg_fov_for_dataset_at_frame(colony, T).squeeze()
        # get segmentation from one timepoint
        relabel = np.zeros_like(img, dtype=np.uint16)

        for track, df_track in df_seg.groupby("track_id"):
            label = int(df_track["label_img"].unique()[0])
            relabel[img == label] = int(df_track["track_id"].unique()[0])

        del img

        OmeTiffWriter().save(relabel, self.output_dir / "TRA" / f"{int(T):04d}.tif")
        return


if __name__ == "__main__":
    fire.Fire(main)
