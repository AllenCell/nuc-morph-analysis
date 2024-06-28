################################################################################
# %%
from pathlib import Path
import os
from nuc_morph_analysis.analyses.inhibitors.match_and_filter import (
    identify_matched_tracks,
    compute_and_normalize_by_initial_volumes,
    filter_to_long_enough_tracks,
    filter_at_death_threshold,
)
from nuc_morph_analysis.analyses.inhibitors import plotting
from nuc_morph_analysis.lib.preprocessing.global_dataset_filtering import load_dataset_with_features
from nuc_morph_analysis.analyses.inhibitors import dataset_info
from nuc_morph_analysis.lib.preprocessing import filter_data, add_features
import numpy as np

# %%

dfo = load_dataset_with_features(dataset="all_drug_perturbation", load_local=True)

pair_dict = dataset_info.drug_analysis_pairs()
# iterate through the drug analysis control-perturb pairs.
pair_list = [
    "aphidicolin_lamin_exp1_controlsONLY",
    "aphidicolin_lamin_exp1_rep2",
    "aphidicolin_lamin_exp2_rep1",
]
for pairs in pair_list:

    print(pairs)
    dict_of_control_perturb_colonies = pair_dict[pairs]._asdict()
    df = dataset_info.preprocess_and_add_columns(dfo, dict_of_control_perturb_colonies)
    details = dataset_info.get_drug_perturbation_details_from_colony_name(
        dict_of_control_perturb_colonies["perturb"]
    )

    # determine folder for saving figures
    savedir = Path(__file__).parent / "figures" / "late_growth" / pairs
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # define mitotic entry differently than default and remove cells that enter or exit mitosis
    df = add_features.add_division_entry_and_exit_annotations(
        df, breakdown_threshold=3, formation_threshold=12
    )
    df = filter_data.filter_out_cells_entering_or_exiting_mitosis(df)
    df = filter_data.all_timepoints_minimal_filtering(df)

    df = compute_and_normalize_by_initial_volumes(df, details["drug_added_frame"])

    # first remove all timepoints at which the number of dead cells exceeds a threshold
    df = filter_at_death_threshold(df, details)

    # now require that tracks minimally start at least 1 hr before drug addition and last at least 2 hr after drug addition
    dff = filter_to_long_enough_tracks(df)

    # now require that each track in the perturbation condition has a matching track in the control condition of equal volume
    p_ids, c_ids = identify_matched_tracks(dff)
    # keep only matched tracks
    dff = dff[dff.track_id.isin(p_ids + c_ids)]

    # export dataframe/parquet here with name something like name = f{pairs}_late_growth_dynamics.parquet

    # now plot the tracks
    for col in ["volume", "volume_sub"]:
        plotting.simple_plot_individual_trajectories(
            dff, col, dict_of_control_perturb_colonies, pairs, matched=True
        )

    for col in ["volume", "volume_sub"]:
        plotting.plot_populations_and_stat_test(
            dff, col, dict_of_control_perturb_colonies, pairs, matched=True
        )
