python linear_regression_workflow.py \
--cols 'volume_at_B','height_at_B','time_at_B','xy_aspect_at_B','SA_vol_ratio_at_B','colony_time_at_B','SA_at_B','mean_height','mean_density','mean_xy_aspect','mean_SA_vol_ratio','mean_neighbor_avg_dxdt_48_volume_whole_colony','std_height','std_density','std_xy_aspect','std_SA_vol_ratio','std_neighbor_avg_dxdt_48_volume_whole_colony' \
--alpha_range 0,0.1,0.5,1,1.3,1.5,2,2.5,5,10,11,12,13 \
--target 'delta_volume_BC' \
--save_path "../../figures/" \
--tolerance 0.05