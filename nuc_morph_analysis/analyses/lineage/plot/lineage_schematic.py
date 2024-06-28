import numpy as np
from pathlib import Path
from nuc_morph_analysis.lib.visualization.movie_tools import (
    create_movie_filestructure,
    create_colorized_fov_movie,
)


def generate_lineage_schematic(df, track_id, timepoints, figdir):
    """
    Plot figures of the a single lineage tree at specified frames

    Paramaters
    ----------
    df: pd.DataFrame
        dataframe with all the timepoints
    figdir: Path
        path to the figure directory
    EXAMPLE_TRACKS: Dict[str, int]
        dictionary with the track ids of the example tracks
    timepoints: List
        list of index_sequence values to plot
    """
    # %% Get family id for nucleus of interest (using cell_id hardcoded below)
    df_track = df.loc[df["track_id"] == track_id]
    family_id = df_track["family_id"].values[0]

    # %% Get only the frames to visualize in the figure (using index_sequence hardcoded below)
    df_sub = df[df["index_sequence"].isin(timepoints)]

    # %% Make a column for the colorizer to use
    tree_of_interest = df_sub.loc[df_sub["family_id"] == family_id, "family_id"]
    df_sub[f"family_id_{family_id}"] = np.nan
    df_sub.loc[tree_of_interest.index, f"family_id_{family_id}"] = tree_of_interest
    variable = f"family_id_{family_id}"

    base_figdir = Path(__file__).parent.parent.parent / figdir

    # %% Create images of frames of interest
    movie_figdir = create_movie_filestructure(base_figdir, [variable], combined_colormap=True)
    create_colorized_fov_movie(
        df=df_sub,
        savedir=movie_figdir,
        var=variable,
        df_all=df_sub,
        colorbar=False,
        cmap=None,
        categorical=True,
        save_type=None,
        save_format="pdf",
        parallel_flag=True,
    )
