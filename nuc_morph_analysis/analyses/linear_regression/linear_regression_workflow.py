import argparse
import ast
import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import (
    RepeatedKFold,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering, filter_data
from sklearn.model_selection import permutation_test_score
from nuc_morph_analysis.lib.visualization.plotting_tools import (
    get_plot_labels_for_metric,
)
import imageio

pd.options.mode.chained_assignment = None  # default='warn'

warnings.simplefilter(action="ignore", category=FutureWarning)


def main(cols, target, alpha_range, tolerance, save_path, cached_dataframe=None):

    save_path = Path(save_path)
    save_path = save_path / Path("linear_regression")
    save_path.mkdir(parents=True, exist_ok=True)

    if not cached_dataframe:
        df_all = global_dataset_filtering.load_dataset_with_features()
        df_full = filter_data.all_timepoints_full_tracks(df_all)
        df_track_level_features = filter_data.track_level_features(df_full)
    else:
        df_track_level_features = pd.read_csv(cached_dataframe)

    fit_linear_regression(
        df_track_level_features, cols, target, alpha_range, tolerance, save_path
    )


def fit_linear_regression(data, cols, target, alpha, tol, save_path):
    """
    data - track level features
    cols - input features
    target - target to predict
    alpha - hyperparameter for lasso
    tol - tolerance to check drop in r^2 for finding best alpha (ex. 0.02)
    save_path - location to save files
    """
    sns.set_context("talk")
    random_state = 2652124

    # init empty dicts and lists
    all_test_sc = []
    all_coef_alpha = []
    all_perms = {
        "score": [],
        "perm_score_mean": [],
        "perm_score_std": [],
        "p_value": [],
        "alpha": [],
    }

    # remove 0 alpha due to convergence errors
    alpha = [i for i in alpha if i != 0]
    alpha = [round(i, 1) for i in alpha]

    # find best alpha for Lasso model
    for alpha_ind, this_alpha in enumerate(alpha):
        print("fitting alpha", this_alpha)

        # drop any nan rows
        dropna_cols = cols + [target]
        data = data.dropna(subset=dropna_cols)

        # make numpy array for inputs and target
        all_input = data[cols].reset_index(drop=True).values
        all_target = data[target].values

        if this_alpha == 0:
            # linear regression if alpha == 0
            clf = linear_model.LinearRegression()
        else:
            clf = linear_model.Lasso(alpha=this_alpha)

        # normalize input features
        model = make_pipeline(StandardScaler(), clf)

        # run permutation test
        score, permutation_scores, pvalue = permutation_test_score(
            model,
            all_input,
            all_target,
            random_state=random_state,
            cv=5,
            n_permutations=500,
        )

        # break if permutation score is less than linear regression value (max possible)
        # with a tolerance
        # or if p_value > 0.05
        rounded_permutation_score = round(score, 2)
        if alpha_ind == 0:
            max_val = rounded_permutation_score
        if abs(rounded_permutation_score - max_val) > tol or (pvalue > 0.05):
            break

        # if relatively equal to linear regression value, then continue
        # save permutation score and p_value to dictionary
        all_perms["score"].append(score)
        all_perms["perm_score_mean"].append(permutation_scores.mean())
        all_perms["perm_score_std"].append(permutation_scores.std())
        all_perms["p_value"].append(pvalue)
        all_perms["alpha"].append(this_alpha)

        # run cross validate to get model coefficients
        cv_model = cross_validate(
            model,
            all_input,
            all_target,
            cv=RepeatedKFold(n_splits=5, n_repeats=20, random_state=random_state),
            return_estimator=True,
            n_jobs=2,
            scoring=[
                "r2",
                "explained_variance",
                "neg_mean_absolute_error",
                "max_error",
                "neg_mean_squared_error",
                "neg_mean_absolute_percentage_error",
            ],
            return_train_score=True,
        )

        # Save test r^2 and test MSE to dataframe
        range_test_scores = [round(i, 2) for i in cv_model["test_r2"]]
        range_errors = [round(i, 2) for i in cv_model["test_neg_mean_squared_error"]]
        test_sc = pd.DataFrame()
        test_sc[r"Test r$^2$"] = range_test_scores
        test_sc["Test MSE"] = range_errors
        test_sc["alpha"] = this_alpha
        all_test_sc.append(test_sc)

        # Save coeffs to dataframe
        coefs = pd.DataFrame(
            [model[1].coef_ for model in cv_model["estimator"]], columns=cols
        )

        coefs["alpha"] = this_alpha
        all_coef_alpha.append(coefs)

    # Get test scores for all alpha
    all_test_sc = pd.concat(all_test_sc, axis=0).reset_index(drop=True)
    all_test_sc["Test MSE"] = -all_test_sc["Test MSE"]
    save_path = save_path / Path(f"{target}")
    save_path.mkdir(parents=True, exist_ok=True)
    all_test_sc.to_csv(save_path / "mse.csv")

    # Get coeffs for all alpha
    all_coef_alpha = pd.concat(all_coef_alpha, axis=0).reset_index(drop=True)
    all_coef_alpha = all_coef_alpha.melt(
        id_vars=["alpha"],
        var_name="Column",
        value_name="Coefficient Importance",
    ).reset_index(drop=True)
    all_coef_alpha.to_csv(save_path / "coefficients.csv")

    # Get permutation scores and p values for all alpha
    all_perms = pd.DataFrame(all_perms).reset_index(drop=True)
    all_perms.to_csv(save_path / "perm_scores.csv")

    # Save coefficient plot for max alpha value
    save_plots(all_coef_alpha, all_test_sc, all_perms, target, save_path)

    return all_coef_alpha, all_test_sc, all_perms


def save_plots(all_coef_alpha, all_test_sc, all_perms, target, save_path):

    xlim = None
    files = []
    for alpha in all_coef_alpha["alpha"].unique():
        this_coef_alpha = all_coef_alpha.loc[
            all_coef_alpha["alpha"] == alpha
        ].reset_index(drop=True)
        this_test_sc = all_test_sc.loc[all_test_sc["alpha"] == alpha].reset_index(
            drop=True
        )
        this_perms = all_perms.loc[all_perms["alpha"] == alpha].reset_index(drop=True)
        p_value = round(this_perms["p_value"].item(), 3)
        test_r2_mean = round(this_test_sc["Test r$^2$"].mean(), 2)
        test_r2_std = round(this_test_sc["Test r$^2$"].std() / 2, 2)

        g = sns.catplot(
            data=this_coef_alpha,
            y="Column",
            x="Coefficient Importance",
            kind="bar",
            errorbar="sd",
            aspect=2,
            height=10,
        )
    
        g.set(ylabel='')
        
        g.fig.subplots_adjust(top=.9)  # adjust the Figure in rp
        g.fig.suptitle(
            f"Prediction of {get_plot_labels_for_metric(target)[1]}\nalpha={alpha}, test r\u00B2={test_r2_mean}±{test_r2_std}, P={p_value}"
        )
        label_list = [
            get_plot_labels_for_metric(col)[1]
            for col in all_coef_alpha["Column"].unique()
        ]
        g.set_yticklabels(label_list)
        print(f"Saving coefficients_{target}_alpha_{alpha}.png")
        this_path = str(save_path / Path(f"coefficients_{target}_alpha_{alpha}.png"))
        files.append(this_path)

        if not xlim:
            xlim = g.fig.axes[0].get_xlim()
        g.set(xlim=xlim)
        g.savefig(this_path, dpi=300)

    # save movie of pngs
    writer = imageio.get_writer(save_path / "coefficients_over_time.mp4", fps=2)
    for im in files:
        writer.append_data(imageio.imread(im))
    writer.close()


def list_of_strings(arg):
    return arg.split(",")


def list_of_floats(arg):
    return list(map(float, arg.split(",")))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False


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
        default=["volume_at_B", "time_at_B", "colony_time_at_B", "SA_at_B"],
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
        default=np.arange(0, 15, 0.1, dtype=float),
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
    args = parser.parse_args()
    main(
        cols=args.cols,
        target=args.target,
        alpha_range=args.alpha_range,
        tolerance=args.tolerance,
        save_path=args.save_path,
        cached_dataframe=args.cached_dataframe,
    )
