################################################################################
# %%
from pathlib import Path
import os
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.lib.preprocessing import filter_data
from nuc_morph_analysis.analyses.inhibitors import dataset_info
from nuc_morph_analysis.analyses.inhibitors import plotting
from nuc_morph_analysis.analyses.inhibitors.match_and_filter import (
    acquire_dividing_cells_before_and_after_drug_addition,
)

# %%
pair_dict = dataset_info.drug_analysis_pairs()

# cannot run "aphidicolin_lamin_exp2_rep1" in this analysis because the perturb colony does not have enough lamin shell formation events
# also puromycin may need to be removed, if analysis is not included in the paper
pair_list = ["importazole_lamin_exp1_rep1"]
dfo = load_dataset_with_features(dataset="all_drug_perturbation")

for pairs in pair_list:
    print(pairs)
    dict_of_control_perturb_colonies = pair_dict[pairs]._asdict()
    for chosen_condition in ["control", "perturb"]:

        df = dataset_info.preprocess_and_add_columns(dfo, dict_of_control_perturb_colonies)
        print(df.colony.unique())
        df = df[df["condition"] == chosen_condition]

        details = dataset_info.get_drug_perturbation_details_from_colony_name(
            dict_of_control_perturb_colonies[chosen_condition]
        )

        # the number of cells dying in perturb details will be used to adjust the x axis for all individual trajectory plots (control and perturb)
        perturb_details = dataset_info.get_drug_perturbation_details_from_colony_name(
            dict_of_control_perturb_colonies["perturb"]
        )

        # determine folder for saving figures
        savedir = f"{Path(__file__).parent}{os.sep}figures{os.sep}expansion{os.sep}{pairs}"
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        df = filter_data.all_timepoints_minimal_filtering(df)

        # define time since frame formation
        df["time_since_frame_formation"] = (
            df["index_sequence"] - df["predicted_formation"]
        ) * details["time_interval_minutes"]

        # export dataframe/parquet here with name something like name = f{pairs}_exapansion_dynamics.parquet

        dfsub1, dfsub2 = acquire_dividing_cells_before_and_after_drug_addition(
            df, details, pairs, chosen_condition
        )

        plotting.expansion_plot_individual(
            df, dfsub1, dfsub2, pairs, chosen_condition, perturb_details
        )

        plotting.expansion_plot_population_avg(
            dfsub1, dfsub2, savedir, pairs, chosen_condition, details
        )

        # now perform bootstrap resampling to determine the ratio and difference between the two conditions
        plotting.bootstrap_ratio_and_difference(dfsub1, dfsub2, pairs, chosen_condition)
