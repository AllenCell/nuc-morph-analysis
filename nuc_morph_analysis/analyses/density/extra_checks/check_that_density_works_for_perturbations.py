# %%
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib
from nuc_morph_analysis.lib.visualization.notebook_tools import save_and_show_plot
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.visualization.reference_points import COLONY_COLORS, COLONY_LABELS
from nuc_morph_analysis.analyses.inhibitors.dataset_info import get_drug_perturbation_details_from_colony_name

matplotlib.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "Arial"

#%%
def try_to_get_drug_name(colony_name):
    try:
        return get_drug_perturbation_details_from_colony_name(colony_name)["drugs_string"]
    except:
        return colony_name

def plot_perturbation_densities(df, figdir, time_axis = 'real_time', error="percentile", show_legend=True, interval=5,titlestr=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    feature_col = "2d_area_nuc_cell_ratio"
    scale, label, units, _ = get_plot_labels_for_metric(feature_col)

    new_colors = plt.cm.tab20(range(20))
    for ci, (colony, df_colony) in enumerate(df.groupby("colony")):
        df_colony = df_colony.sort_values("index_sequence")

        color = COLONY_COLORS.get(colony,new_colors[ci])

        if time_axis == "real_time":
            time_col = "index_sequence"
            x_label = "Real Time (hr)"
        if time_axis == "colony_time":
            time_col = "colony_time"
            x_label = "Aligned Colony Time (hr)"
        
        grouper = df_colony[[time_col] + [feature_col]].groupby(time_col)[
                feature_col
            ]
        
        # filter grouper so that only timepoints with more than 15 cells are included
        count = grouper.count()
        log_count = count[count>15].index

        
        mean_density = grouper.mean() * scale
        if error == "std":
            std_density = grouper.std() * scale
            lower = mean_density - std_density
            upper = mean_density + std_density
        if error == "percentile":
            lower = grouper.quantile(0.05) * scale
            upper = grouper.quantile(0.95) * scale

        time = mean_density.index.values * interval / 60
        
        time = time[log_count]
        mean_density = mean_density[log_count]
        lower = lower[log_count]
        upper = upper[log_count]

        ax.fill_between(
            time,
            lower,
            upper,
            alpha=0.12,
            color=color,
            zorder=0,
            edgecolor="none",
            label=COLONY_LABELS.get(colony,try_to_get_drug_name(colony)),
        )
        ax.plot(
            time, mean_density, linewidth=1.2, color=color, label="", zorder=20
        )

    ax.set_ylabel(f"Average Density \n Across Colony {units}")
    ax.set_xlabel(x_label)
    if show_legend is True:
        # ax.legend(loc="upper right", handletextpad=0.7, frameon=False)
        # put legend outside to the right
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), frameon=False)
    plt.title(titlestr)
    plt.tight_layout()
    # save_and_show_plot(
    #     f"{figdir}/avg_density_colony_{time_axis}_alignment-{feature_col}",
    #     file_extension=".pdf",
    #     dpi=300,
    #     transparent=True,
    # )

from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
for dataset in ["all_drug_perturbation","all_feeding_control","all_baseline"]:
    df0 = load_dataset_with_features(dataset,load_local=True)
    df = filter_data.all_timepoints_minimal_filtering(df0)
    figdir = f"figures/{dataset}/density_plots"
    plot_perturbation_densities(df, figdir, time_axis = 'real_time', error="percentile", show_legend=True, interval=5,titlestr=dataset)


#%%