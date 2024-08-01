# %%
from pathlib import Path
import pandas as pd
import numpy as np
from nuc_morph_analysis.lib.preprocessing.add_features import add_SA_vol_ratio, add_aspect_ratio
import matplotlib.transforms as transforms
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from sklearn import metrics
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
import re

from nuc_morph_analysis.lib.preprocessing import all_datasets

# define the figure save directory
save_dir = Path(__file__).parent / "figures" / "quantitative_validation_plots"
if not save_dir.exists():
    save_dir.mkdir(exist_ok=True, parents=True)

# define parameters for plotting
fontsize = 8
markersize = 10
dpi = 300
figsize = (1.2, 1.2)
file_extension = ".pdf"
transparent = True

# %%
csv_url = all_datasets.segmentation_model_validation_URLs["single_cell_features"]
df0 = pd.read_csv(csv_url)

# %%
# perform PCA on the spherical harmonic coefficients
n_components = 8
feature_columns = [x for x in df0.columns.tolist() if "shcoeffs" in x]

# determine the PCA fit on the watershed segmentations only
dfin = df0.loc[df0.Name == "watershed_segmentation", :]
target_df = dfin
pca = PCA(n_components=n_components)
pca.fit(target_df[feature_columns].values)

# now apply fitted PCA transform to all columns in the dataframe (watershed and model segmentations)
target_df = df0
pca_out = pca.transform(target_df[feature_columns].values)
dfpca = pd.DataFrame(pca_out)
PCA_columns = [f"PC{i+1}" for i in range(n_components)]
# dfpca.columns = PCA_columns
dfpca.rename(columns={i: col for i, col in enumerate(PCA_columns)}, inplace=True)
dfpca["CellId"] = target_df["CellId"]
dfpca["Name"] = target_df["Name"]

df0 = df0.merge(
    dfpca[["CellId", "Name"] + PCA_columns],
    on=["CellId", "Name"],
    how="left",
    suffixes=("", "_pca"),
)
# %%
# add SA/Vol ratio
df0 = add_SA_vol_ratio(df0)
# add aspect ratios
df0 = add_aspect_ratio(df0)

df = df0.set_index(["CellId", "Name"]).unstack(level=1)
# %%
figdir = save_dir


ycol = "model_segmentation"
xcol = "watershed_segmentation"  #
feature_list = [
    "volume",
    "mesh_sa",
    "SA_vol_ratio",
    "xy_aspect",
    "xz_aspect",
    "zy_aspect",
    "height",
]
feature_list.extend([f"PC{i}" for i in range(1, 9)])


feat_list = []
for feature in feature_list:
    scale_factor, label, unit, _ = get_plot_labels_for_metric(feature)
    xy = df[feature][[xcol, ycol]].dropna()
    xyv = xy.values * scale_factor
    x = xy[xcol] * scale_factor
    y = xy[ycol] * scale_factor
    error = y - x

    # quantify metrics
    feats = {"feature": feature}
    r2 = metrics.r2_score(x, y)
    rmse = metrics.mean_squared_error(x, y, squared=False)
    meanerror = np.mean(error)
    nvalue = xy.shape[0]

    # compute percent bias
    interpercentile_range = x.quantile(0.95) - x.quantile(0.05)
    percent_bias = 100 * meanerror / interpercentile_range

    # compute percent_se (or variance-normalized rmse)
    percent_se = 100 * rmse / interpercentile_range

    feats.update(
        {
            "r2": r2,
            "rmse": rmse,
            "meanerror": meanerror,
            "percent_bias": percent_bias,
            "nvalue": nvalue,
            "percent_se": percent_se,
        }
    )
    feat_list.append(pd.DataFrame(data=feats.values(), index=feats.keys()).T)

    # create scatter figure
    file_name = f"{feature}_scatter"
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.scatter(
        x,
        y,
        s=markersize,
        marker="o",
        facecolors="None",
        edgecolors="k",
        linewidths=0.5,
        zorder=200,
    )
    ax.set_aspect("equal", "box")
    ax.axline(
        (xyv.min(), xyv.min()),
        (xyv.max(), xyv.max()),
        color="k",
        linestyle="--",
        linewidth=0.85,
        zorder=100,
        label="y=x",
    )

    # set labels
    ax.set_xlabel("100x lamin segmentation", fontsize=fontsize)
    ax.set_ylabel("model predicted\nsegmentation", fontsize=fontsize)
    ax.set_title(f"Prediction accuracy\n{label} {unit}", fontsize=fontsize)
    leg = ax.legend(loc="lower right", fontsize=fontsize)
    leg.set_alpha(0)
    leg.get_frame().set_linewidth(0.0)

    # now adjust axes
    ax.margins(0.1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.tick_params("both", labelsize=fontsize)

    # add metrics to the plot
    textlist = [
        f'RMSE={rmse:.2f} {unit.replace("(","").replace(")","")}',
        f"R2={r2:.2f}",
        f"N={nvalue}",
    ]
    ax.text(
        0.05,
        0.98,
        "\n".join(textlist),
        transform=ax.transAxes,
        fontsize=fontsize - 1,
        ha="left",
        va="top",
    )
    save_and_show_plot(
        f'{figdir}/{re.sub("[#/()]", "", file_name)}',
        file_extension=file_extension,
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close()

    # create histogram figure
    file_name = f"{feature}_error_hist"
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    bins = np.linspace(error.min(), error.max(), 30)
    if (bins[1] - bins[0] < scale_factor) & ("(" in unit) & ("ratio" not in feature):
        print(f"resizing bins for {feature}")
        bins = np.arange(error.min(), error.max() + scale_factor, scale_factor)
    ax.hist(error, bins=bins, color="gray", density=True)
    ax.axvline(meanerror, color="k", linestyle="--", label="mean error", linewidth=0.85)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.set_xlabel(f"Error {unit}\n(predicted - ground truth)", fontsize=fontsize)
    ax.set_title(f"Error distribution\n{label} {unit}", fontsize=fontsize)

    # now adjust axes
    ax.margins(x=0.1, y=0.3)
    # make the limits of the X axis symmetrics, so zero is always in center
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3, symmetric=True))
    xlim = ax.get_xlim()
    maxs = np.abs(xlim).max() * np.array([-1, 1])
    mins = np.abs(xlim).max() * np.array([1, -1])
    lims = np.stack([maxs, mins])
    ax.set_xlim(lims.min(), lims.max())
    # now ylims, etc
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.tick_params("both", labelsize=fontsize)

    # make the text for the mean error occur to the right of the meanerror line

    textlist = [
        f"%bias=\n{percent_bias:.2f} %",
        f"Avg.=\n{meanerror:.2f}",
        f"N={nvalue}",
    ]
    ha = "left" if meanerror > 0 else "right"
    xloc = 0.05 if meanerror > 0 else 0.95

    ax.text(
        xloc,
        0.98,
        "\n".join(textlist),
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=fontsize - 1,
    )

    save_and_show_plot(
        f'{figdir}/{re.sub("[#/()]", "", file_name)}',
        file_extension=file_extension,
        bbox_inches="tight",
        transparent=transparent,
    )
    plt.close()  # close plot after saving to avoid opening too many figure windows.

# %%
df_feats = pd.concat(feat_list)
df_feats = df_feats.iloc[::-1]

# create bar plot figure
statistic_list = ["r2", "percent_bias", "rmse", "meanerror", "percent_se"]
for statistic in statistic_list:
    file_name = f"{statistic}_bar"
    fig, ax = plt.subplots(1, 1, figsize=(1, 0.2 * df_feats.shape[0]), dpi=dpi)
    width = df_feats[statistic].values
    xy = df[feature][[xcol, ycol]].dropna()
    xyv = xy.values * scale_factor
    y0 = df_feats["feature"].values
    y = [get_plot_labels_for_metric(f)[1] for f in y0]
    ax.barh(y, width, left=None, align="center", color="gray", edgecolor="k", linewidth=0.5)
    # now print the actual value within the bar
    text_list = []
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, v in enumerate(width):
        text_list.append(
            ax.text(1.02, i, f"{v:.2f}", ha="left", va="center", fontsize=fontsize, transform=trans)
        )

    ax.tick_params("both", labelsize=fontsize)
    ax.set_title(f"{statistic} values", fontsize=fontsize)
    ax.set_xlabel(f"{statistic}", fontsize=fontsize)
    xticks = [0, 0.5, 1] if statistic == "r2" else [-50, 0, 50]
    ax.set_xticks([0, 0.5, 1])
    if statistic in ["percent_se"]:
        ax.set_xticks([0, 25, 50])

    if statistic in ["percent_bias", "meanerror"]:
        ax.axvline(0, color="k", linestyle="-", linewidth=0.7)
        # make the limits of the X axis symmetrics, so zero is always in center
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3, symmetric=True))
        xlim = ax.get_xlim()
        maxs = np.abs(xlim).max() * np.array([-1, 1])
        mins = np.abs(xlim).max() * np.array([1, -1])
        lims = np.stack([maxs, mins])
        lims_min, lims_max = lims.min(), lims.max()
        lim_range = lims_max - lims_min
        ax.set_xlim(lims_min - lim_range * 0, lims_max + lim_range * 0)

    save_and_show_plot(
        f'{figdir}/{re.sub("[#/()]", "", file_name)}',
        file_extension=file_extension,
        bbox_inches="tight",
        transparent=transparent,
    )
# %%
