import matplotlib.pyplot as plt
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot

FONTSIZE = 20


def feature_over_time(df, pixel_size, interval, feature_type, tid=None):
    """
    Plot how the fixed features fluctuate over "time". The changes seen are due to segmentation error NOT changes in the cell.

    Paramaters
    ----------
    df: dataframe
        fixed control dataframe
    pixel_size: float
        pixel size returned by load_data.get_dataset_pixel_size('fixed_control')
    interval: float
        interval returned by load_data.get_dataset_time_interval_in_min('fixed_control')
    feature_type: string
        one of 'height', 'surface_area', or 'volume'
    tid: int
        track id
    ploterror: boolean
        if True, plots error bounds

    Returns
    -------
    figure
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for _, dft in df.groupby("track_id"):
        x = dft.index_sequence.to_numpy()
        if feature_type == "height":
            y = dft.height.to_numpy() * pixel_size
            plt.ylabel("Height (µm)", fontsize=FONTSIZE)
        elif feature_type == "surface_area":
            y = dft.mesh_sa.to_numpy() * pixel_size * pixel_size
            plt.ylabel("Surface Area (µm²)", fontsize=FONTSIZE)
        elif feature_type == "volume":
            y = dft.volume.to_numpy() * pixel_size * pixel_size * pixel_size
            plt.ylabel("Volume (µm³)", fontsize=FONTSIZE)
        else:
            print("Invalid feature type")
            return
        plt.plot(x - x.min(), y, alpha=0.3)

    if tid != None:
        dft = df[df["track_id"] == tid]
        x = dft.index_sequence.to_numpy()
        if feature_type == "height":
            y = dft.height.to_numpy() * pixel_size
        elif feature_type == "surface_area":
            y = dft.mesh_sa.to_numpy() * pixel_size * pixel_size
        elif feature_type == "volume":
            y = dft.volume.to_numpy() * pixel_size * pixel_size * pixel_size
        ax.plot(x, y, color="#000000", label=tid)
        plt.legend(loc="upper left")

    plt.xticks([0, 10, 20])
    plt.xlabel("Frames", fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.tight_layout()
    save_and_show_plot(f"error_morflowgenesis/figures/feature_over_time_{feature_type}_{tid}")


def shape_modes_over_time(df, interval, tid=None):
    """
    Plot shape modes over "time". Fluctuations seen could be due to segmentation error. However, these errors may lead to more
    extreme chnages in shape modes due to changes in the longest axis and therefore their alignment prior to PCA.

    Paramaters
    ----------
    df: dataframe
        fixed control dataframe
    interval: float
        interval returned by load_data.get_dataset_time_interval_in_min('fixed_control')
    tid: int
        track id

    Returns
    -------
    figure
    """
    shape_modes = [
        "NUC_PC1",
        "NUC_PC2",
        "NUC_PC3",
        "NUC_PC4",
        "NUC_PC5",
        "NUC_PC6",
        "NUC_PC7",
        "NUC_PC8",
    ]
    shape_modes_title = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"]

    for i in range(len(shape_modes)):
        fig, ax = plt.subplots(figsize=(5, 5))
        for _, dft in df.groupby("track_id"):
            y = dft[shape_modes[i]]
            x = dft.index_sequence.to_numpy() * interval
            plt.plot(x, y, alpha=0.3)

        if tid != None:
            dft = df[df["track_id"] == tid]
            y = dft[shape_modes[i]]
            x = dft.index_sequence.to_numpy() * interval
            plt.plot(x, y, color="#000000", label=tid)
            plt.legend(loc="upper left")

        plt.xlabel("Time (min)", fontsize=FONTSIZE)
        plt.ylabel(shape_modes_title[i], fontsize=FONTSIZE)
        plt.xticks([0, 50, 100])
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)
        plt.tight_layout()
        save_and_show_plot(
            f"error_morflowgenesis/figures/shapemode_over_time_{shape_modes[i]}_{tid}"
        )
