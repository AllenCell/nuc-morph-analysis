# %%
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering


def run_inhibitor_work():
    from nuc_morph_analysis.analyses.inhibitors import (
        expansion_dynamics_workflow,
        late_growth_dynamics_workflow,
    )

    expansion_dynamics_workflow
    late_growth_dynamics_workflow


_ = global_dataset_filtering.load_dataset_with_features("all_drug_perturbation", save_local=True)

run_inhibitor_work()
