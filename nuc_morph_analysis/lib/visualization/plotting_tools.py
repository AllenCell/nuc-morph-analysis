from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d

from nuc_morph_analysis.lib.visualization.label_tables import (
    get_scale_factor_table,
    get_one_to_one_dict,
    LABEL_TABLE,
    UNIT_TABLE,
    LIMIT_TABLE,
    COLORIZER_LABEL_TABLE,
)
from nuc_morph_analysis.lib.preprocessing.load_data import (
    get_dataset_time_interval_in_min,
)
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
from nuc_morph_analysis.utilities.analysis_utilities import get_correlation_values


def get_plot_labels_dict(dataset="all_baseline", colorizer=False):
    """
    Combines scale factor, label and unit table into a single dict

    Returns
    ----------
    plot_labels_dict: dict
        dictionary with key: (scale, label, unit)
    """
    scale_factor_table = get_scale_factor_table(dataset=dataset)
    scale_factor_dict = get_one_to_one_dict(scale_factor_table)
    if colorizer:
        label_dict = get_one_to_one_dict(COLORIZER_LABEL_TABLE)
    else:
        label_dict = get_one_to_one_dict(LABEL_TABLE)
    unit_dict = get_one_to_one_dict(UNIT_TABLE)
    limit_dict = get_one_to_one_dict(LIMIT_TABLE)
    dict_list = [scale_factor_dict, label_dict, unit_dict, limit_dict]
    plot_labels_dict = {}

    for mapping_dict in dict_list:
        for key in mapping_dict:
            if key not in plot_labels_dict:
                plot_labels_dict[key] = (
                    scale_factor_dict.get(key, 1),
                    label_dict.get(key, key),
                    unit_dict.get(key, ""),
                    limit_dict.get(key, None),
                )

    return plot_labels_dict


def get_plot_labels_for_metric(metric_name, dataset="all_baseline", colorizer=False):
    """
    Returns the scale_factor, label, unit and limits for a metric

    Parameters
    ----------
    metric_name: str
        name of the metric to obtain plot labels for

    Returns
    ----------
    plot_labels: tuple
        scale factor, label, unit, limits for the input metric
    """
    plot_labels_dict = get_plot_labels_dict(dataset=dataset, colorizer=colorizer)

    for key, plot_labels in plot_labels_dict.items():
        if key.lower() == metric_name.lower():
            return plot_labels

    return (1, metric_name, "", None)


def clean_up_label(label_string):
    """
    Cleans up label_string to be usable as a file name.
    Changes `label_string` to lower case and replaces spaces with underscores.
    Also removes units specified within regular brackets

    Parameters
    ----------
    label_string: string
        Input string. E.g.: "Volume at formation"

    Returns
    ----------
    clean_label: string
        Output string. E.g.: "volume_at_formation"

    """
    if label_string.find("(") >= 0:
        label_string = label_string[: (label_string.find("("))]
    for _, value in get_plot_labels_dict().items():
        unit_str = value[-2]
        label_string = label_string.lower().replace(unit_str, "")
    clean_label = label_string.lower().strip().replace(" ", "_")
    return clean_label


def create_scatter_plot(
    metric_x,
    metric_y,
    label_x,
    label_y,
    save_dir=None,
    title=None,
    correlation_metric="pearson",
    save_format="png",
    plot_linear_fit=True,
    scatter_args=None,
    plot_args=None,
    bootstrap_count=0,
    y_lim=None,
    use_slope=False,
):
    """
    Creates a scatter plot from input metrics, and returns the figure and axes

    Parameters
    ----------
    metric_x, metric_y: Lists, Numpy arrays of the same size
        X and Y metrics to use to create the scatter plot
    label_x, label_y: string
        Labels to use on the x and y axes
    save_dir: path, string
        Directory to save scatter plot to
    title: string
        Title of the scatter plot
    correlation_metric: string
        Metric to use to calculate correlation between x and y metrics
    save_format: string
        Format to save the scatter plot in
    plot_linear_fit: bool
        Flag for whether to apply and plot a linear fit to the data
    scatter_args: dict
        Arguments to pass to the scatter plot
    plot_args: dict
        Arguments to pass to the plot
    bootstrap_count: int
        Number of bootstrap iterations to use for error estimation
    y_lim: array
        Array containing the lower and upper limits for the y axis
    Returns
    ----------
    fig, ax: matplotlib objects
        figure and axis objects containing the scatter plot
    """
    if scatter_args is None:
        scatter_args = {
            "s": 20,
            "alpha": 0.75,
            "facecolor": "gray",
            "edgecolor": "none",
        }
    if plot_args is None:
        plot_args = {
            "color": "black",
            "linewidth": 1,
        }
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)

    coefficient, p_value, err_val, ci = get_correlation_values(
        metric_x, metric_y, correlation_metric, bootstrap_count
    )

    if ci is not None:
        corr_label = (
            f"N={len(metric_x)}, " f"Corr={coefficient:.2g}, " f"({ci[0]:.2g}, {ci[1]:.2g})"
        )
    elif use_slope:
        slope = coefficient * np.nanstd(metric_y) / np.nanstd(metric_x)
        r2 = coefficient**2
        corr_label = f"N={len(metric_x)}, " f"Slope={slope:.2g}, " f"r\u00B2={r2:.2g}"
    else:
        corr_label = f"N={len(metric_x)}, " f"Corr={coefficient:.2g}, " f"p-value: {p_value:.2g}"

    ax.scatter(metric_x, metric_y, **scatter_args)

    if plot_linear_fit:
        linear_fit = np.polyfit(metric_x, metric_y, 1)
        predict_func = np.poly1d(linear_fit)
        ax.plot(
            metric_x,
            predict_func(metric_x),
            label=corr_label,
            **plot_args,
        )
    ax.set(xlabel=label_x, ylabel=label_y)
    if title is None:
        title = corr_label
    else:
        title += f"\n{corr_label}"
    ax.set_title(title)
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    plt.tight_layout()

    if save_dir is not None:
        if "/" in label_y:
            label_y = label_y.replace("/", "divby")
        filename_x = clean_up_label(label_x)
        filename_y = clean_up_label(label_y)
        filename = f"scatter_{filename_y}_vs_{filename_x}.{save_format}"
        filename = filename.replace("\n", "")
        fig.savefig(
            f"{save_dir}/{filename}",
            facecolor="w",
            format=save_format,
        )
        plt.show()

    return fig, ax, (coefficient, p_value, err_val, ci)


def get_filtered_values_from_df_column(
    df, col_to_plot, time_col, scale_factor=1, smooth_frames=0, pmin=0, pmax=100
):
    """
    This is a general purpose function that returns filtered and smoothened values
    from a dataframe column. This is useful while dropping values outside a certain
    percentile range, and also for smoothening the values.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe containing the data to plot
    col_to_plot: str
        name of the column containing the data to plot
    time_col: str
        name of the column containing the time values
    scale_factor: float
        factor to scale the y values by
    smooth_frames: int
        number of frames to use for smoothening
    pmin: float
        minimum percentile to use for filtering
    pmax: float
        maximum percentile to use for filtering

    Returns
    ----------
    time_vals: np.array
        array of filtered time values
    y_vals: np.array
        array of filtered y values
    df_filtered: pd.DataFrame
        filtered dataframe
    """
    vals_nonan = df[col_to_plot].dropna().values
    vmin, vmax = np.percentile(vals_nonan, [pmin, pmax])
    df_filtered = df.loc[(df[col_to_plot] >= vmin) & (df[col_to_plot] <= vmax)]
    df_filtered = df_filtered.sort_values(time_col)
    y_vals = df_filtered.groupby(time_col)[col_to_plot].agg("mean").values * scale_factor
    time_vals = df_filtered[time_col].unique()

    if smooth_frames:
        y_vals = gaussian_filter1d(y_vals, smooth_frames)

    return time_vals, y_vals, df_filtered


def add_error_patch_to_axis(
    df,
    time_col,
    err_cols,
    time_vals,
    y_vals,
    ax,
    title_str,
    y_scale_factor=1,
    color="C0",
):
    """
    Adds error patch to axis

    Parameters
    ----------
    df_dataset: pd.DataFrame
        dataframe containing the data to plot
    time_col: str
        name of the column containing the time values
    err_cols: list
        list of column names containing the error values
    time_vals: np.array
        array of filtered time values
    y_vals: np.array
        array of filtered y values
    ax: matplotlib axis object
        axis to add the error patch to
    title_str: str
        string to add to the title
    y_scale_factor: float
        factor to scale the y values by
    color: str
        color to use for the error patch

    Returns
    ----------
    ax: matplotlib axis object
        axis with the error patch added
    title_str: str
        updated title string
    lower_bound: np.array
        array of lower bound values
    upper_bound: np.array
        array of upper bound values
    """
    if len(err_cols) == 1:
        err_col = err_cols[0]
        err_vals = df.groupby(time_col)[err_col].agg("mean").values * y_scale_factor
        lower_bound = y_vals - err_vals
        upper_bound = y_vals + err_vals
        title_str += " mean $\pm$ std. err."
    elif len(err_cols) == 2:
        err_col_lower, err_col_upper = err_cols
        lower_bound = df.groupby(time_col)[err_col_lower].agg("mean").values * y_scale_factor
        upper_bound = df.groupby(time_col)[err_col_upper].agg("mean").values * y_scale_factor
        title_str += " mean $\pm$ 95% CI"
    else:
        raise ValueError("err_cols must have length 1 or 2")

    ax.fill_between(time_vals, lower_bound, upper_bound, alpha=0.3, color=color, edgecolor="none")

    return ax, title_str, lower_bound, upper_bound


def plot_time_series_by_frame(
    df,
    col_to_plot="height",
    err_cols=None,
    time_col="index_sequence",
    save_dir=None,
    smooth_frames=0,
    align_to_colony_time=False,
    name_prefix="",
    xlabel=None,
    ylabel=None,
    axs=None,
    additional_col_to_plot=None,
    save_format="png",
):
    """
    Plots the average column value by frame vs time for each dataset

    Parameters
    ----------
    df: dataframe
        dataframe containing dataset/colony column
    col_to_plot: str
        column to be plotted
    err_cols: list
        list of columns to be used for error bars
        If the list has two elements, the first element is used for the lower error bar
        and the second element is used for the upper error bar
        If the list has one element, the same element is used for both error bars
    save_dir: path
        path at which images are saved
    smooth_frames: int
        number of frames to use for smoothening
    align_to_colony_time: bool
        creates additional plot with superimposed traces aligned to colony time
    name_prefix: str
        additional string to be added before column name while saving
    xlabel: str
        label for x axis
    ylabel: str
        label for y axis
    axs: matplotlib axes
        axes to be used for plotting
    """
    num_dataset = df["colony"].nunique()

    if err_cols is None:
        err_cols = []

    if axs is None:
        fig, axs = plt.subplots(1, num_dataset, figsize=(5 * num_dataset, 4), dpi=300, sharey=True)
    else:
        fig = axs[0].get_figure()

    if num_dataset == 1:
        axs = [axs]

    df = df.dropna(subset=col_to_plot).copy()
    y_scale_factor, label, unit, _ = get_plot_labels_for_metric(col_to_plot)

    ylabel = ylabel if ylabel is not None else f"{label} {unit}"

    if additional_col_to_plot is not None:
        ax_t = [ax.twinx() for ax in axs]

    if align_to_colony_time:
        fig_a, ax_a = plt.subplots(1, 1, figsize=(5, 4), dpi=300)

    for ac, (dataset, df_dataset) in enumerate(df.groupby("colony")):
        time_vals, y_vals, df_dataset = get_filtered_values_from_df_column(
            df_dataset, col_to_plot, time_col, y_scale_factor, smooth_frames
        )
        title_str = COLONY_LABELS[dataset]

        axs[ac].plot(time_vals, y_vals, label=COLONY_LABELS[dataset], c=COLONY_COLORS[dataset])

        axs[ac].axhline(y=0, color="black", linestyle="--")

        if len(err_cols):
            axs[ac], title_str, lower_bound, upper_bound = add_error_patch_to_axis(
                df_dataset,
                time_col,
                err_cols,
                time_vals,
                y_vals,
                axs[ac],
                title_str,
                y_scale_factor,
                color=COLONY_COLORS[dataset],
            )

        axs[ac].set_title(title_str)

        if align_to_colony_time:
            if "colony_time" not in df_dataset.columns:
                aligned_time_vals = time_vals
            else:
                (
                    aligned_time_vals,
                    aligned_y_vals,
                    df_dataset,
                ) = get_filtered_values_from_df_column(
                    df_dataset,
                    col_to_plot,
                    "colony_time",
                    y_scale_factor,
                    smooth_frames,
                )

            ax_a.plot(
                aligned_time_vals,
                aligned_y_vals,
                label=COLONY_LABELS[dataset],
                c=COLONY_COLORS[dataset],
            )

            ax_a.axhline(y=0, color="black", linestyle="--")

            if len(err_cols):
                ax_a.fill_between(
                    aligned_time_vals,
                    lower_bound,
                    upper_bound,
                    alpha=0.3,
                    label="_no_legend_",
                    color=COLONY_COLORS[dataset],
                    edgecolor="none",
                )

        if additional_col_to_plot is not None:
            add_scale_factor, add_label, add_unit, _ = get_plot_labels_for_metric(
                additional_col_to_plot
            )
            (
                t_vals_additional,
                y_vals_additional,
                df_dataset,
            ) = get_filtered_values_from_df_column(
                df_dataset, additional_col_to_plot, time_col, add_scale_factor
            )
            ax_t[ac].plot(
                t_vals_additional,
                y_vals_additional,
                label=add_label,
                c=COLONY_COLORS[dataset],
                linestyle="--",
            )

    fig.supxlabel(xlabel if xlabel is not None else "Time (frames)")
    axs[0].set_ylabel(ylabel)

    if additional_col_to_plot is not None:
        ax_t[-1].set_ylabel(f"{add_label} {add_unit}")

    if align_to_colony_time:
        ax_a.legend()
        ax_a.set_xlabel("Colony time (frames)")
        ax_a.set_ylabel(ylabel)

    plt.tight_layout()
    if save_dir is not None:
        save_name = f"{name_prefix}{col_to_plot}"
        if smooth_frames:
            save_name += "_smoothened"
        fig.savefig(save_dir / f"{save_name}.{save_format}", facecolor="w", format=save_format)

        if align_to_colony_time:
            save_name = f"{name_prefix}{col_to_plot}_vs_colony_time"
            if smooth_frames:
                save_name += "_smoothened"
            fig_a.savefig(
                save_dir / f"{save_name}.{save_format}",
                facecolor="w",
                format=save_format,
            )
        plt.show()

    if align_to_colony_time:
        return fig, axs, fig_a, ax_a

    return fig, axs


def normalize_values(values):
    """
    Normalizes values to be between 0 and 1

    Parameters
    ----------
    values: np.array
        array of values

    Returns
    ----------
    values: np.array
        normalized array of values
    """
    if values.max() != values.min():
        values = (values - values.min()) / (values.max() - values.min())
    elif values.max() == values.min():
        values = np.ones(values.shape)
    return values


def plot_weighted_scatterplot_by_frame(
    df,
    col_to_plot="height",
    time_col="index_sequence",
    save_dir=None,
    align_to_colony_time=False,
    name_prefix="",
    xlabel=None,
    ylabel=None,
    weights_col=None,
    plot_scale=1,
    axs=None,
    additional_col=None,
    save_format="png",
    min_size=1,
):
    """
    Plots the average column value by frame vs time for each dataset

    Parameters
    ----------
    df: dataframe
        dataframe containing dataset/colony column
    col_to_plot: str
        column to be plotted (contains y positions)
    save_dir: path
        path at which images are saved
    smooth_weight: float between 0 and 1
        weight for smoothing. 0 = disable
    align_to_colony_time: bool
        creates additional plot with superimposed traces aligned to colony time
    name_prefix: str
        additional string to be added before column name while saving
    """

    num_dataset = df["colony"].nunique()

    if axs is None:
        fig, axs = plt.subplots(1, num_dataset, figsize=(6 * num_dataset, 4), dpi=300, sharey=True)
    else:
        fig = axs[0].get_figure()

    if num_dataset == 1:
        axs = [axs]

    df = df.dropna(subset=col_to_plot).copy()
    scale_factor, label, unit, _ = get_plot_labels_for_metric(col_to_plot)

    ylabel = ylabel if ylabel is not None else f"{label} {unit}"

    if additional_col is not None:
        ax_t = [ax.twinx() for ax in axs]

    if align_to_colony_time:
        fig_a, ax_a = plt.subplots(1, 1, figsize=(6, 4), dpi=300)

    all_legend_elements = []
    for ac, (dataset, df_dataset) in enumerate(df.groupby("colony")):
        time_vals, y_vals, df_dataset = get_filtered_values_from_df_column(
            df_dataset, col_to_plot, time_col, scale_factor
        )

        if weights_col is not None:
            raw_weights = df_dataset.groupby(time_col)[weights_col].agg("mean").values ** 2
        else:
            raw_weights = np.ones(y_vals.shape)

        # normalize weights
        weights = normalize_values(raw_weights)
        sizes = min_size + weights * plot_scale

        axs[ac].scatter(
            time_vals,
            y_vals,
            s=sizes,
            label=COLONY_LABELS[dataset],
            facecolor=COLONY_COLORS[dataset],
            edgecolor="none",
            alpha=0.7,
        )

        axs[ac].axhline(y=0, color="black", linestyle="--")

        if weights_col is not None:
            legend_elements = [
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    fc="w",
                    fill=False,
                    edgecolor="none",
                    linewidth=0,
                    label=COLONY_LABELS[dataset],
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=COLONY_COLORS[dataset],
                    label=f"min. r\u00B2: {raw_weights.min():.3f}",
                    markerfacecolor=COLONY_COLORS[dataset],
                    markersize=np.sqrt(sizes.min()),
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=COLONY_COLORS[dataset],
                    label=f"max. r\u00B2: {raw_weights.max():.3f}",
                    markerfacecolor=COLONY_COLORS[dataset],
                    markersize=np.sqrt(sizes.max()),
                ),
            ]
            all_legend_elements.extend(legend_elements)
            axs[ac].legend(handles=legend_elements)
            axs[ac].set_title(COLONY_LABELS[dataset])

        if align_to_colony_time:
            if "colony_time" not in df_dataset.columns:
                aligned_time_vals = time_vals
                aligned_y_vals = y_vals
            else:
                (
                    aligned_time_vals,
                    aligned_y_vals,
                    df_dataset,
                ) = get_filtered_values_from_df_column(
                    df_dataset,
                    col_to_plot,
                    "colony_time",
                    scale_factor,
                )
            if weights_col is not None:
                aligned_raw_weights = np.abs(
                    df_dataset.groupby(time_col)[weights_col].agg("mean").values
                )
            else:
                aligned_raw_weights = np.ones(y_vals.shape)
            aligned_weights = normalize_values(aligned_raw_weights)
            sizes = min_size + aligned_weights * plot_scale
            aligned_time_vals_hr = [
                frame * get_dataset_time_interval_in_min(dataset) / 60
                for frame in aligned_time_vals
            ]

            ax_a.scatter(
                aligned_time_vals_hr,
                aligned_y_vals,
                s=sizes,
                label=COLONY_LABELS[dataset],
                facecolor=COLONY_COLORS[dataset],
                edgecolor="none",
                alpha=0.7,
            )
            if weights_col is not None:
                ax_a.legend(
                    handles=all_legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.4),
                    ncol=3,
                    columnspacing=0.75,
                )
            ax_a.axhline(y=0, color="black", linestyle="--")

        if additional_col is not None:
            add_scale_factor, add_label, add_unit, _ = get_plot_labels_for_metric(additional_col)

            (
                t_vals_additional,
                y_vals_additional,
                df_dataset,
            ) = get_filtered_values_from_df_column(
                df_dataset, additional_col, time_col, add_scale_factor
            )

            ax_t[ac].plot(
                t_vals_additional,
                y_vals_additional,
                label=add_label,
                c=f"C{ac}",
                linestyle="--",
            )

    fig.supxlabel(xlabel if xlabel is not None else "Time (frames)")
    axs[0].set_ylabel(ylabel)

    if additional_col is not None:
        ax_t[-1].set_ylabel(f"{add_label} {add_unit}")

    if align_to_colony_time:
        ax_a.set_xlabel("Aligned colony time (hr)")
        ax_a.set_ylabel(ylabel)

    plt.tight_layout()
    if save_dir is not None:
        save_name = f"{name_prefix}{col_to_plot}"
        fig.savefig(save_dir / f"{save_name}.{save_format}", facecolor="w", format=save_format)

        if align_to_colony_time:
            save_name = f"{name_prefix}{col_to_plot}_vs_colony_time"
            fig_a.savefig(
                save_dir / f"{save_name}.{save_format}",
                facecolor="w",
                format=save_format,
            )
        plt.show()

    if align_to_colony_time:
        return fig, axs, fig_a, ax_a

    return fig, axs

def colorize_image(mip, dft, feature='2d_area_nuc_cell_ratio'):
    """
    Function create an image where the segmentation image objects (e.g. nuclei)
    are colored by the a given feature from the dataframe of that timepoint 

    Parameters
    ----------
    mip : np.array
        the max intensity projection of the labeled image
    dft : pd.DataFrame
        the dataframe of the timepoint
    feature : str
        the feature to color the image by
    """
    
    # now recolor the image by matching the pixel values in image to label_img in dft
    recolored_img = np.zeros_like(mip).astype('float32')
    recolored_img[mip>0]=np.nan
    for _,row in dft.iterrows():
        recolored_img[mip==row['label_img']] = row[feature]
    return recolored_img