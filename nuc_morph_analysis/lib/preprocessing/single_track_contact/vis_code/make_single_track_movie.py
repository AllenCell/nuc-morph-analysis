# %%
from tqdm import tqdm
from multiprocessing import Pool

from nuc_morph_analysis.lib.preprocessing.single_track_contact.export_code.export_helper import (
    get_single_track_contact_save_dir,
    define_track_save_paths,
    process_track,
    load_exported_fov_imgs,
)
from nuc_morph_analysis.lib.preprocessing.load_data import get_dataset_names
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features

FEATURE_COLUMNS = [
    "volume",
    "mesh_sa",
    "SA_vol_ratio",
    "height",
    "xz_aspect",
    "xy_aspect",
    "zy_aspect",
    "colony_depth",
    "neigh_distance",
    "density",
    "dxdt_48_volume",
    "dxdt_24_volume",
    "dxdt_12_volume",
]
NECESSARY_COLUMNS = [
    "CellId",
    "track_id",
    "index_sequence",
    "roi",
    "Ff",
    "frame_transition",
    "Fb",
    "is_outlier",
    "fov_edge",
]


# %%
def make_single_track_movie(
    num_workers=32,
    full_tracks_only=True,
    overwrite=False,
    dataset=None,
    single_frame=False,
    track_id_list=None,
    testing=False,
):
    """
    Export 8bit MIPs for all datasets in the 20x EGFP channel
    Exports a full size MIP and a 2x2 downsampled MIP

    Parameters
    ----------
    num_workers : int, optional
        Number of workers to use for parallel processing, by default 32
    full_tracks_only : bool, optional
        Whether to only export full tracks, by default True
    overwrite : bool, optional
        Whether to overwrite existing files, by default False
    dataset : str, optional
        Name of the dataset to process, by default None
    single_frame : bool, optional
        Whether to export only a single_frame, by default False
    track_id_list : list, optional
        List of track_ids to process, by default None
    testing : bool, optional
        Whether to run in testing mode, by default False
    """
    dataset_names = get_dataset_names(dataset)

    for dataset in dataset_names:
        # retreive the dataframe with the tracking info
        df0 = load_dataset_with_features(dataset)

        dffov = df0.groupby("index_sequence").agg("first").reset_index()

        df = df0.copy()
        del df0
        if full_tracks_only == True:
            df = df[df["is_full_track"] == True]

        df = df[NECESSARY_COLUMNS + FEATURE_COLUMNS]
        # set all feature columns to be float
        for col in FEATURE_COLUMNS:
            df[col] = df[col].astype(float)
        for col in ["is_outlier", "fov_edge"]:
            df[col] = df[col].astype(bool)

        track_id_list = df["track_id"].unique()
        print(f"Number of tracks to save: {len(track_id_list)}")
        save_dir = get_single_track_contact_save_dir(
            dataset, create=True, single_frame=single_frame
        )
        print("Save directory: ", save_dir)
        if single_frame:
            print("Exporting only a single frame")

        # now ask which tracks have already been saved
        # determine which ones already exist and which need to be skipped
        if overwrite == False:
            track_id_list2 = [
                tid
                for tid in track_id_list
                if define_track_save_paths(dataset, tid, save_dir).exists() == False
            ]
            print(
                "skipping ",
                len(track_id_list) - len(track_id_list2),
                " tracks that have already been saved",
            )
            track_id_list = track_id_list2

        # preload the fov_frames
        fov_images = load_exported_fov_imgs(dataset, dffov.index_sequence.unique(), small_fov=True)

        # args_list = [(df[df['track_id'] == tid], fov_images, dffov, dataset, save_dir, num_workers, single_frame) for tid in track_id_list]
        args_list = [
            (
                df[df["track_id"] == tid],
                fov_images,
                dffov,
                dataset,
                save_dir,
                num_workers,
                single_frame,
            )
            for tid in track_id_list
        ]

        if testing:
            args_list = args_list[:num_workers]
            overwrite = True

        if num_workers == 1:
            with tqdm(total=len(args_list)) as mainbar:
                for args in args_list:
                    process_track_wrapper(args)
                    mainbar.update(1)
        else:
            print("running in parallel")
            with Pool(num_workers) as p:
                with tqdm(total=len(args_list)) as mainbar:
                    for i, _ in enumerate(p.imap_unordered(process_track_wrapper, args_list)):
                        mainbar.update(1)


def process_track_wrapper(args):
    dft, fov_images, dffov, dataset, save_dir, num_workers, single_frame = args
    process_track(dft, fov_images, dffov, dataset, save_dir, num_workers, single_frame=single_frame)
    # now save the tif_stack as movie
    # movie function


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg_or_raw", type=str, default="raw", help="Whether to export the raw or seg channel"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers to use for parallel processing",
    )
    parser.add_argument(
        "--full_tracks_only", type=str, default="True", help="Whether to only export full tracks"
    )
    parser.add_argument(
        "--overwrite", type=str, default="False", help="Whether to overwrite existing files"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Name of the dataset to process")
    parser.add_argument(
        "--single_frame", type=str, default="False", help="Whether to export only a single_frame"
    )
    parser.add_argument(
        "--track_id_list", type=list, default=None, help="List of track_ids to process"
    )
    parser.add_argument(
        "--testing", type=str, default="False", help="Whether to run in testing mode"
    )
    args = parser.parse_args()

    # True
    args.overwrite = args.overwrite.lower() in ("true", "1", "yes")
    args.full_tracks_only = args.full_tracks_only.lower() in ("true", "1", "yes")
    args.single_frame = args.single_frame.lower() in ("true", "1", "yes")
    args.testing = args.testing.lower() in ("true", "1", "yes")

    make_single_track_movie(
        num_workers=args.num_workers,
        full_tracks_only=args.full_tracks_only,
        overwrite=args.overwrite,
        dataset=args.dataset,
        single_frame=args.single_frame,
        track_id_list=args.track_id_list,
        testing=args.testing,
    )
