python linear_regression_workflow_greedy_removal.py \
--cached_dataframe "/allen/aics/modeling/ritvik/projects/trash/nucmorph/nuc-morph-analysis/track_level.csv" \
--target 'duration_BC' \
--alpha_range "0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2" \
--save_path "/allen/aics/modeling/ritvik/projects/trash/nucmorph/nuc-morph-analysis/nuc_morph_analysis/figures_debug/" \
--tolerance 0.08 \
--max_iterations 4