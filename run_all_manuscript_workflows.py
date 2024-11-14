import argparse
from nuc_morph_analysis.utilities.workflow_runner import get_jobs, execute
from nuc_morph_analysis.analyses.colony_area import colony_area_workflow


class Workflows:
    def figure_1_dataset():
        import nuc_morph_analysis.analyses.dataset_images_for_figures.figure_1_workflow
        # panel E items are generated in segmentation_model_validation
        colony_area_workflow.baseline_colony_starting_area()
        
    def figure_s1_cell_health():
        import nuc_morph_analysis.analyses.cell_health.figure_s1_workflow
        
    def figure_s2_segmentation_model_validation(): # and s3??
        from nuc_morph_analysis.analyses.segmentation_model_validation import (
            seg_model_validation_figure_workflow,
        )
        seg_model_validation_figure_workflow.save_out_specified_image_pairs_with_overlays()
        import nuc_morph_analysis.analyses.segmentation_model_validation.quantitative_validation_workflow
        
    # figure 2 images generated using timelapse feature exploroer   
            
    def figure_3_s4_height_density():
        # figure 3 images generated using timelapse feauture explorer
        import nuc_morph_analysis.analyses.height.figure_3_s4_workflow
        
    # figure s5 is raw image data can be found on quilt
    
    def figure_s6_inhibitors():
        import nuc_morph_analysis.analyses.inhibitors.figure_s6_workflow
        colony_area_workflow.aphidicolin_control_starting_area()
        colony_area_workflow.importazole_control_starting_area()
        
    def figure_s7_growth_outliers():
        import nuc_morph_analysis.analyses.evaluate_filters_and_outliers.figure_s7_workflow
    
    def figure_4_s8_volume_trajectories():
        import nuc_morph_analysis.analyses.volume.figure_4_s8_workflow
        
    def figure_5_and_s9_local_growth():
        import nuc_morph_analysis.analyses.volume.figure_5_s9_workflow
        #s10 coming soon

    def figure_6_s11_compensation():
        import nuc_morph_analysis.analyses.volume_variation.figure_6_s11_workflow
        
    def figure_s12_feeding_control():
        import nuc_morph_analysis.analyses.feeding_control.figure_s12_workflow
        colony_area_workflow.feeding_control_starting_area()

    def figure_7_s13_s14_lineage():
        import nuc_morph_analysis.analyses.lineage.figure_7_s13_s14_workflow

    def figure_s15_linear_regression_model():
        import nuc_morph_analysis.analyses.linear_regression.figure_s15_workflow
    
    # figure s16 is this in s6?    
    # figure s17, s18 ? none?
    
    def figure_s19_transition_point():
        import nuc_morph_analysis.analyses.volume.figure_s19_workflow
        
    def figure_s20_precision_error():
        import nuc_morph_analysis.analyses.error_morflowgenesis.workflows.figure_s20_workflow
    
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
