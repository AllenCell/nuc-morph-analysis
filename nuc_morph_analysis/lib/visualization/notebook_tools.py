import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"


def save_and_show_plot(
    filename,
    file_extension=".pdf",
    figure=None,
    transparent=True,
    quiet=False,
    massive_output=False,
    remove_all_points_in_pdf=False,
    keep_open=False,
    **kwargs,
):
    """
    Use this instead of matplotlib.pyplot.show()
    Render a plot in a notebook (if applicable) and save the plot to a file

    Parameters
    ----------
    filename: String
        Relative path within nuc_morph_analysis/analyses folder to save the figure to. The figures
        should go in a figures/ directory in the same folder as the workflow that generates them.

    file_extension: String, optional
        The file extension. Default '.png'

    figure: matplotlib.figure.Figure or maplotlib.figure.SubFigure, optional
        If provided, the figure to save. If omitted this function will find figures with the global
        pyplot interface.

    transparent: boolean
        Flag to save figure with a transparent background

    quiet: boolean
        Flag to suppress print statements

    massive_output: boolean
        Flag to allow saving >1000 figures at once

    remove_all_points_in_pdf: boolean
        Flag to remove all points in the pdf. This removes all scatter plot points and just saves the axes.
        it also saves a png version of the file

    keep_open: boolean
        Flag to keep the figure open after saving

    kwargs: dict, optional
        This function takes any number of keyword arguments and passes them to savefig. See
        matplotlib.pyplot.savefig documentation for the full list of options.

    Example
    -------
    save_and_show_plot("colony_context/figures/aggregate_colony_positions/all_aggregate_colony_positions", dpi=300, facecolor="w")
    """
    # We expect the string "/figures/" to be in filename
    if not re.search("/figures/", filename):
        raise Exception(
            "Figures should be written to a figures/ directory in same folder as the workflow"
        )

    if figure is None:
        num_figs = len(plt.get_fignums())
        if num_figs > 1000 and not massive_output:
            raise RuntimeError(
                f"Attempted to write {num_figs} figures."
                "If you really want to save lots of output files, use the massive_output=True flag."
            )
        fig_num_list = list(plt.get_fignums())
    else:
        fig_num_list = [figure.number]

    for i in fig_num_list:
        # If multiple figures have been plotted, they get names like correlation1.png,
        # correlation2.png, correlation3.png
        this_figure = plt.figure(i)
        num_str = str(i) if len(fig_num_list) > 1 else ""

        if remove_all_points_in_pdf:
            # save pdf
            save_fig(
                filename + file_extension,
                this_figure,
                transparent=transparent,
                quiet=quiet,
                **kwargs,
            )
            # it also saves a png version of the file
            save_fig(filename + ".png", this_figure, transparent=transparent, quiet=quiet, **kwargs)

            # Remove all points in the pdf. This removes all scatter plot points and just saves the axes.
            ax = this_figure.gca()
            # remove all scatter objects
            for coll in ax.collections:
                coll.remove()
            # save with objects removed
            save_fig(
                filename + "_removed" + file_extension,
                this_figure,
                transparent=transparent,
                quiet=quiet,
                **kwargs,
            )
        else:
            save_fig(
                filename + num_str + file_extension,
                this_figure,
                transparent=transparent,
                quiet=quiet,
                **kwargs,
            )
    if not keep_open:
        plt.show()
        plt.close()


def save_and_show_correlation_plot(
    filename_prefix,
    relationship_axis,
    feature,
    generations,
    control,
    distance_threshold=None,
    volume_threshold=None,
    **kwargs,
):
    """
    Render a plot in a notebook (if applicable) and save the plot to a file. Use the simpler
    save_and_show_plot if the arguments here are not applicable.


    Parameters
    ----------
    filename_prefix: String
        Relative path within nuc_morph_analysis/analyses folder. Must include '/figures/'. May
        include an initial prefix of the filename. E.g. 'lineage/figures/multi_generation'

    relationship_axis: String
        'width' for sisters, 'depth' for mother-daughter

    generations: Int Array
        List of generation numbers included

    outliers: Boolean
        True if outliers are excluded

    control: Boolean
       True if dataframe is a control dataframe of unrelated pairs

    kwargs: dict, optional
        This function takes any number of keyword arguments and passes them to save_and_show_plot
        (and ultimately to matplotlib.pyplot.savefig).
    """
    relationship = "mother-daughter" if relationship_axis == "depth" else "sisters"
    generations_str = ",".join([str(generations)])
    control_str = (
        f"_control_distancethresh{distance_threshold}_volthresh{volume_threshold}"
        if control
        else ""
    )
    save_and_show_plot(
        f"{filename_prefix}{feature}_{relationship}_generations{generations_str}{control_str}",
        **kwargs,
    )


def create_figures_dir(path):
    """
    Create figures folder in nuc-morph-analysis/nuc_morph_analysis/analysis
    as specified in path.

    Parameters
    ----------
    path: str
        Relative path where figures will be saved. For example:
        figures/my_analysis/
    """
    base_dir = Path(__file__).parent.parent.parent  # Path to nuc-morph-analysis/nuc_morph_analysis
    figs_dir = Path.joinpath(base_dir, "analyses", path)
    os.makedirs(figs_dir, exist_ok=True)
    return figs_dir


def save_fig(filename, figure, transparent=True, quiet=False, **kwargs):
    """
    E.g. save_fig("volume/figures/some_plot.pdf", figure)
    """
    figs_dir = create_figures_dir(path=Path(filename).parent)
    absolute_filename = figs_dir / Path(filename).name
    figure.savefig(absolute_filename, transparent=transparent, **kwargs)
    if not quiet:
        print(f"Wrote figure {absolute_filename}")
