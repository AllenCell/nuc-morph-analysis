global_dataset_filtering functions
add_volume_change_over_25_minute_window --> 
    'volume_change_over_25_minutes'

filter_out_dips.run_script(df_full) -->
    
    'volume_dip_peak_mask_at_region', # boolean array, true at all points within peak region(s)
    'volume_dip_peak_mask_at_center', # boolean array, true at all peak centers
    'volume_dip_has_peak', # boolean array, true at all points if there is a peak
    'volume_dip_volume_change_at_center', # magnitude value at each peak center (left_base - peak)
    'volume_dip_volume_change_at_region', # magnitude values at all points within peak region(s) (left_base - peak)
    'volume_dip_width_at_center', # width of the peak at the peak center index
    'volume_dip_width_at_region', # width of the peak at all indices in the peak region
    'volume_dip_max_volume_change', # maximum magnitude value (at all points in array)
    'volume_dip_peak_id_at_center', # peak id at the peak center index
    'volume_dip_peak_id_at_region', # peak id at all indices in the peak region
    'volume_dip_total_number', # total number of peaks
    "volume_dip_removed_um_unfilled", # dips or jumps removed (refilled with nans)
    
    'volume_dips_removed_um_unfilled' # volume trajectories with peak (volume dip) regions removed

then compute growth rate to get -->
    'dxdt_48_volume_dips_removed_um_unfilled'

and compute neighbors to get -->
    'neighbor_avg_dxdt_48_volume_dips_removed_um_unfilled_90um'
    'neighbor_avg_dxdt_48_volume_dips_removed_um_unfilled_whole_colony'
