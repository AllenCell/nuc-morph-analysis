import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from nuc_morph_analysis.analyses.linear_regression.select_features import (
    get_feature_list,
)
from nuc_morph_analysis.analyses.linear_regression.linear_regression_workflow import (
    fit_linear_regression,
)
from nuc_morph_analysis.analyses.linear_regression.utils import (
    list_of_strings,
    list_of_floats,
)
from nuc_morph_analysis.lib.visualization.plotting_tools import (
    get_plot_labels_for_metric,
)
import imageio
import seaborn as sns
import os
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(
    cols,
    target,
    alpha_range,
    tolerance,
    save_path,
    max_iterations,
    cached_dataframe=None,
):

    save_path = Path(save_path)
    save_path = save_path / Path("linear_regression_greedy") / Path(f"{target}")
    save_path.mkdir(parents=True, exist_ok=True)

    if len(cols) < 1:
        cols = get_feature_list(["features"], target)

    label_list = [get_plot_labels_for_metric(col)[1] for col in cols]
    map_dict = {i: j for i, j in zip(cols, label_list)}

    if not cached_dataframe:
        df_all = global_dataset_filtering.load_dataset_with_features()
        df_full = filter_data.all_timepoints_full_tracks(df_all)
        df_track_level_features = filter_data.track_level_features(df_full)
        df_track_level_features.to_csv(
            "/allen/aics/modeling/ritvik/projects/trash/nucmorph/nuc-morph-analysis/track_level.csv"
        )
    else:
        df_track_level_features = pd.read_csv(cached_dataframe)

    permute_cols = []
    count = 0

    removal_coefs = []
    removal_test_sc = []
    removal_perms = []
    while (count < len(cols) - 2) and (count < max_iterations):
        print(
            f"Iteration {count}, Total cols {len(cols)}, Removing features {permute_cols}"
        )
        try:
            all_coef_alpha, all_test_sc, all_perms = fit_linear_regression(
                df_track_level_features,
                cols,
                target,
                alpha_range,
                tolerance,
                save_path,
                False,
                permute_cols,
            )
        except ValueError:
            break
        # Save max alpha
        max_alpha = all_coef_alpha["alpha"].max()
        all_coef_alpha = all_coef_alpha.loc[all_coef_alpha["alpha"] == max_alpha]
        all_test_sc = all_test_sc.loc[all_test_sc["alpha"] == max_alpha]
        all_perms = all_perms.loc[all_perms["alpha"] == max_alpha]
        all_coef_alpha["iteration"] = count
        all_test_sc["iteration"] = count
        all_perms["iteration"] = count

        # Get max coefficient to remove
        tmp = all_coef_alpha.copy()
        tmp["Coefficient Importance"] = tmp["Coefficient Importance"].abs()
        tmp = tmp.loc[
            tmp["Coefficient Importance"] == tmp["Coefficient Importance"].max()
        ]
        remove_column = tmp["Column"].item()

        # add this to list of columns to permute
        permute_cols.append(remove_column)

        # save out info
        all_coef_alpha["iteration"] = count
        all_test_sc["iteration"] = count
        all_perms["iteration"] = count

        all_coef_alpha["feature_removed"] = remove_column
        all_test_sc["feature_removed"] = remove_column
        all_perms["feature_removed"] = remove_column

        all_coef_alpha = all_coef_alpha.groupby(["Column"]).mean().reset_index()

        label_list = [
            get_plot_labels_for_metric(col)[1]
            for col in all_coef_alpha["Column"].unique()
        ]
        all_coef_alpha["Column"] = label_list

        all_coef_alpha.to_csv(save_path / f"removal_coefficients_{count}.csv")
        all_test_sc.to_csv(save_path / f"removal_test_sc_{count}.csv")
        all_perms.to_csv(save_path / f"removal_perms_{count}.csv")

        removal_coefs.append(all_coef_alpha)
        removal_test_sc.append(all_test_sc)
        removal_perms.append(all_perms)

        count += 1

    removal_coefs = pd.concat(removal_coefs, axis=0).reset_index(drop=True)
    removal_test_sc = pd.concat(removal_test_sc, axis=0).reset_index(drop=True)
    removal_perms = pd.concat(removal_perms, axis=0).reset_index(drop=True)

    removal_coefs.to_csv(save_path / "removal_coefficients.csv")
    removal_test_sc.to_csv(save_path / "removal_test_sc.csv")
    removal_perms.to_csv(save_path / "removal_perms.csv")

    save_plots(
        removal_coefs, removal_test_sc, removal_perms, target, save_path, map_dict
    )


def save_plots(all_coef_alpha, all_test_sc, all_perms, target, save_path, map_dict):
    sns.set_style("whitegrid")
    xlim = None
    files = []
    perm_cols = []
    perm_coeffs = []
    y_order = []

    xlim = [
        all_coef_alpha["Coefficient Importance"].min(),
        all_coef_alpha["Coefficient Importance"].max(),
    ]
    for iter in all_coef_alpha["iteration"].unique():
        this_coef_alpha = all_coef_alpha.loc[
            all_coef_alpha["iteration"] == iter
        ].reset_index(drop=True)
        this_test_sc = all_test_sc.loc[all_test_sc["iteration"] == iter].reset_index(
            drop=True
        )
        this_perms = all_perms.loc[all_perms["iteration"] == iter].reset_index(
            drop=True
        )
        if iter > 0:
            prev_perms = all_perms.loc[all_perms["iteration"] == iter - 1].reset_index(
                drop=True
            )
            prev_coefs = all_coef_alpha.loc[
                all_coef_alpha["iteration"] == iter - 1
            ].reset_index(drop=True)
            feat_removed = map_dict[prev_perms["feature_removed"].item()]
            feat_coefficient = prev_coefs.loc[prev_coefs["Column"] == feat_removed][
                "Coefficient Importance"
            ].item()
            perm_cols.append(feat_removed)
            perm_coeffs.append(feat_coefficient)
        print(perm_cols, perm_coeffs)
        p_value = round(this_perms["p_value"].item(), 3)
        test_r2_mean = round(this_test_sc["Test r$^2$"].mean(), 2)
        test_r2_std = round(this_test_sc["Test r$^2$"].std() / 2, 2)

        this_coef_alpha["removed"] = False
        if len(perm_cols) > 0:
            this1 = this_coef_alpha.loc[this_coef_alpha["Column"].isin(perm_cols)]
            this2 = this_coef_alpha.loc[~this_coef_alpha["Column"].isin(perm_cols)]
            for col in perm_cols:
                ind = perm_cols.index(col)
                final_coeff = perm_coeffs[ind]
                this1.loc[this1["Column"] == col, "Coefficient Importance"] = (
                    final_coeff
                )
            this1["removed"] = True
            this_coef_alpha = pd.concat([this1, this2], axis=0).reset_index(drop=True)

        if len(y_order) == 0:
            y_order = this_coef_alpha["Column"].values

        g = sns.catplot(
            data=this_coef_alpha,
            y="Column",
            x="Coefficient Importance",
            hue="removed",
            kind="bar",
            order=y_order,
            errorbar="sd",
            aspect=2,
            height=10,
        )

        g.set(ylabel="")

        g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
        g.fig.suptitle(
            f"Prediction of {get_plot_labels_for_metric(target)[1]}\niteration={iter}, test r\u00B2={test_r2_mean}±{test_r2_std}, P={p_value}"
        )
        label_list = [col for col in all_coef_alpha["Column"].unique()]
        g.set_yticklabels(label_list)
        plt.grid()
        this_path = str(save_path / Path(f"coefficients_{target}_iteration_{iter}.png"))
        files.append(this_path)

        g.set(xlim=xlim)
        g.savefig(this_path, dpi=300)

    # save movie of pngs
    writer = imageio.get_writer(save_path / "coefficients_over_time.mp4", fps=2)
    for im in files:
        writer.append_data(imageio.imread(im))
        os.remove(im)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the linear regression workflow")
    # Optional command line argument
    parser.add_argument(
        "--cached_dataframe",
        type=str,
        metavar="path",
        help="Supply a path to a dataframe to skip data preprocessing. If included, dataframe "
        "should match the result of linear_regression_analysis.get_data (see source code for "
        "details).",
    )

    parser.add_argument(
        "--cols",
        type=list_of_strings,
        default=[],
        help="Supply a list of column names to use as independent variables in the linear regression analysis.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="duration_BC",
        help="Supply a column name for a dependent variable to perform regression on",
    )
    parser.add_argument(
        "--alpha_range",
        type=list_of_floats,
        default=np.arange(0.5, 15, 0.2, dtype=float),
        help="Supply a list of alpha values to use in lasso regression",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="figures",
        help="local folder name where plots will be saved",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Tolerace for change in regression score to determine best alpha",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Max iterations for greedy removal",
    )
    args = parser.parse_args()
    main(
        cols=args.cols,
        target=args.target,
        alpha_range=args.alpha_range,
        tolerance=args.tolerance,
        save_path=args.save_path,
        max_iterations=args.max_iterations,
        cached_dataframe=args.cached_dataframe,
    )
