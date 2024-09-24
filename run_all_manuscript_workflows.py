import argparse
from nuc_morph_analysis.utilities.workflow_runner import get_jobs, execute
from nuc_morph_analysis.analyses.colony_area import colony_area_workflow


class Workflows:
    def figure1_dataset():
        import nuc_morph_analysis.analyses.dataset_images_for_figures.run_all_figure_1_code

        # panel E items are generated in segmentation_model_validation

    def figure1_main_text():
        colony_area_workflow.baseline_colony_starting_area()

    def figure3_figureS2_height():
        import nuc_morph_analysis.analyses.height.height_workflow

    def figure4_figureS5_volume_trajectories():
        # excludes aphidicolin panel S5C
        import nuc_morph_analysis.analyses.volume.trajectories_workflow

    def figure5_figureS6_local_growth():
        import nuc_morph_analysis.analyses.volume.local_growth_workflow

    def figure6_figureS7_compensation():
        import nuc_morph_analysis.analyses.volume_variation.compensation_workflow

    def figure7_lineage():
        import nuc_morph_analysis.analyses.lineage.lineage_manuscript_workflow

    def figure_inhibitors():
        import nuc_morph_analysis.analyses.inhibitors.run_all_inhibitor_analysis_workflow

        colony_area_workflow.aphidicolin_control_starting_area()
        colony_area_workflow.importazole_control_starting_area()

    def transition_point():
        import nuc_morph_analysis.analyses.volume.supplement_vis_transition_point_calculation_workflow

    def growth_outliers():
        import nuc_morph_analysis.analyses.evaluate_filters_and_outliers.growth_feature_outliers_workflow

    def error():
        import nuc_morph_analysis.analyses.error_morflowgenesis.workflows.fixed_control_analysis_manuscript_workflow

    def feeding_control():
        import nuc_morph_analysis.analyses.feeding_control.feeding_control_manuscript_workflow

        colony_area_workflow.feeding_control_starting_area()

    def figureS1_segmentation_model_validation():
        from nuc_morph_analysis.analyses.segmentation_model_validation import (
            seg_model_validation_figure_workflow,
        )

        seg_model_validation_figure_workflow.save_out_specified_image_pairs_with_overlays()
        import nuc_morph_analysis.analyses.segmentation_model_validation.quantitative_validation_workflow
        
    def supplemental_figure_cell_health():
        import nuc_morph_analysis.analyses.cell_health.cell_health_workflow


ALL_WORKFLOWS = get_jobs(Workflows)

parser = argparse.ArgumentParser(description="Run all manuscript workflows")

# Optional command line argument
parser.add_argument(
    "--only",
    default=[],
    nargs="+",
    help="Only run the specified workflows. Separate names with spaces.",
)
parser.add_argument(
    "--list",
    action="store_true",
    default=False,
    help="List available workflows.",
)

args = parser.parse_args()
only = args.only

workflows = ALL_WORKFLOWS
if args.list:
    for wf in workflows:
        print(wf.__name__)
else:
    if len(only) >= 1:
        workflows = [wf for wf in workflows if wf.__name__ in only]
    execute(workflows, verbose=True)
