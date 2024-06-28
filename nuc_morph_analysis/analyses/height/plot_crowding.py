import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from nuc_morph_analysis.lib.visualization.movie_tools import get_colorized_img
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot


def get_axis_limits_from_points(points, padding=250):
    """
    Get the axis limits for a plot based on a set of points.

    Parameters
    ----------
    points : numpy.ndarray
        An array of shape (N, 2) representing the x and y coordinates of the points.
    padding : int, optional
        The amount of padding to add to the axis limits (default is 250).

    Returns
    -------
    xlim : list
        A list containing the minimum and maximum x-axis limits.
    ylim : list
        A list containing the minimum and maximum y-axis limits.
    """
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    xlim = [min_x - padding, max_x + padding]
    ylim = [min_y - padding, max_y + padding]
    return xlim, ylim


def plot_density_schematic(
    df_timepoint, track_centroid, neighbor_centroids, frame_centroids, pix_size, figdir
):
    """
    Plot the schematic showing how density is calculated

    Parameters
    ----------
    df_timepoint : DataFrame
        The timepoint data containing information about the nuclei.
    track_centroid : tuple
        The centroid coordinates of the tracked nucleus.
    neighbor_centroids : array-like
        The centroid coordinates of the neighboring nuclei.
    frame_centroids : array-like
        The centroid coordinates of all nuclei in the frame.
    pix_size : float
        The pixel size in micrometers.
    figdir : str
        The directory to save the generated figure.

    Returns
    -------
    None
        This function does not return anything. The figure is saved in the specified directory.
    """
    img = get_colorized_img(df_timepoint, "index_sequence", 1, filter_vals=False)
    fig, axs = plt.subplots(1, 2, figsize=(8, 5), dpi=300)
    for ct, ax in enumerate(axs):
        ax.imshow(img, cmap="Greys_r", origin="upper")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

        # draw arrows to each neighbor
        for centroid in neighbor_centroids:
            ax.arrow(
                track_centroid[0],
                track_centroid[1],
                centroid[0] - track_centroid[0],
                centroid[1] - track_centroid[1],
                head_width=10,
                head_length=10,
                fc="blue",
                ec="blue",
            )

        if ct == 0:  # plot entire colony with all nuclei
            xlim, ylim = get_axis_limits_from_points(frame_centroids, padding=250)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim[1], ylim[0])
            sz = 10
            scale = 50
            size_vertical = 30
            scale_bar_y_pos = 0.02
        else:  # plot zoomed in view of nucleus and neighbors
            avg_neighbor_distance = (
                np.mean(np.linalg.norm(neighbor_centroids - track_centroid, axis=1)) * pix_size
            )
            density = 1 / avg_neighbor_distance**2
            xlim, ylim = get_axis_limits_from_points(neighbor_centroids, padding=50)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim[1], ylim[0])
            sz = 50
            scale = 10
            size_vertical = 6
            title_str = (
                f"Mean distance to neighbors $(r)$: {avg_neighbor_distance:.2f}$\\mu m$"
                + "\n"
                + f"Density $(1 / r ^2)$: {density:.2g}/$\\mu m^2$"
            )
            scale_bar_y_pos = -0.1

            ax.set_title(title_str)

        # create scalebar
        scalebar = AnchoredSizeBar(
            ax.transData,
            scale / pix_size,
            f"{scale} $\\mu m$",
            "lower right",
            bbox_to_anchor=(0.95, scale_bar_y_pos),
            bbox_transform=ax.transAxes,
            color="black",
            pad=0,
            borderpad=0,
            frameon=False,
            size_vertical=size_vertical,
            label_top=False,
        )
        ax.add_artist(scalebar)

        # draw nucleus centroid
        ax.scatter(track_centroid[0], track_centroid[1], color="red", s=sz)

    plt.tight_layout(pad=0)
    save_and_show_plot(f"{figdir}/density_schematic", figure=fig, bbox_inches="tight")
