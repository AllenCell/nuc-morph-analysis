    
|Columns in manifest|Definition|Referenced as in manuscript and timelapse feature explorer|Units|Computed by|
|---|---|---|---|---|
|CellId|Unique identifier for one nucleus at one timepoint, corresponding to each row in the manifest.||unitless|[cyto-dl](https://github.com/AllenCellModeling/cyto-dl/)|
|label_img|Object ID unique for each nucleus in a given FOV.||unitless|cyto-dl|
|track_id|Unique identifier for one nucleus tracked over time.|Track ID|unitless|[aics-timelapse-tracking](https://github.com/AllenCell/aics-timelapse-tracking/) and [nuc-morph-analysis](https://github.com/AllenCell/nuc-morph-analysis)|
|colony|Unique identifier for each timelapse.|Small, Medium, Large|unitless|[nuc-morph-analysis](https://github.com/AllenCell/nuc-morph-analysis)|
|index_sequence|Frame number of the timelapse acquisition. The first frame of the movie is 0. The interval of the movie is used to convert this metric to time.|Time|frames|morflowgenesis|
|roi|The 3D bounding box region around the nucleus segmentation [bottom z, top z, bottom y, top y, bottom x, top x].||pixels|morflowgenesis|
|centroid_x|Horizontal x position of the nucleus segmentation's centroid relative to the field of view.||pixels|morflowgenesis|
|centroid_y|Horizontal y position of the nucleus segmentation's centroid relative to the field of view.||pixels|morflowgenesis|
|centroid_z|Vertical position of the nucleus segmentation's centroid relative to the field of view. The bottom of the FOV is 0.||pixels|morflowgenesis|
|volume|Number of voxels in the nucleus segmentation. This column is used to calculate volume in the paper when scaled by the pixel size cubed.|Volume|pixels cubed|morflowgenesis|
|height|Distance between the pixels with the first and 99th percentile Z-position values within the nucleus segmentation.  This column is used to calculate height in the paper when scaled by the pixel size squared.|Height|pixels|morflowgenesis|
|mesh_vol|Number of voxels in the smoothed mesh of the nuclear segmentation.||pixels cubed|morflowgenesis|
|mesh_sa|Number of pixels on the surface of the smoothed mesh of the nuclear segmentation. This column is used to calculate surface area in the paper when scaled by the pixel size squared.|Surface area|pixels squared|morflowgenesis|
|SA_vol_ratio|The ratio of mesh_sa / volume. Can be used when scaled by the pixel size^2 / pixel size^3.|Surface area to volume ratio|1 / pixels|nuc-morph-analysis|
|transform_params|The aisshparams transform parameters (x, y and z coordinates of the nuclear mesh centroid and the xy angle) used to do 2D alignment to the longest axis prior to spherical harmonic calculation.||feature dependent|morflowgenesis|
|NUC_|Columns that begin with NUC_ are single nucleus shape features and spherical harmonic coefficients. See [cvapipe_analysis](https://github.com/AllenCell/cvapipe_analysis/) for more details.||feature dependent|morflowgenesis|
|length|The length of the longest XY axis of the nuclear segmentation.||pixels|morflowgenesis|
|width|The width of the nuclear segmentation perpendicular to the XY axis.||pixels|morflowgenesis|
|xz_aspect|The ratio of length / height.|XZ aspect ratio|unitless|nuc-morph-analysis|
|xy_aspect|The ratio of length / width.|XY aspect ratio|unitless|nuc-morph-analysis|
|zy_aspect|The ratio of height / width.|ZY aspect ratio|unitless|nuc-morph-analysis|
|fov_edge|Is true if nuclear segmentation touches the edge of the field of view. Nuclear segmentations that touch the FOV edge are not analyzed in the paper.||boolean|cyto-dl|
|predicted_formation|Index sequence of lamin shell formation for a given nuclear trajectory predicted by the interphase detector model.||frames|cyto-dl|
|predicted_breakdown|Index sequence of lamin shell breakdown for a given nuclear trajectory predicted by the interphase detector model.||frames|cyto-dl|
|Ff|Index sequence of lamin shell formation used in the paper. Occurs at a frame where a segmentation is present, not an outlier, and within two frames of the predicted index sequence.||frames|nuc-morph-analysis|
|Fb|Index sequence of lamin shell breakdown used in the paper. Occurs at a frame where a segmentation is present, not an outlier, and within two frames of the predicted index sequence.||frames|nuc-morph-analysis|
|after_breakdown_outlier|Timepoint occurs after Ff.||boolean|nuc-morph-analysis|
|before_formation_outlier|Timepoint occurs before Fb.||boolean|nuc-morph-analysis|
|is_after_breakdown_before_formation_outlier|Is true if the timepoint occurs before frame formation or after frame breakdown. If true, the timepoint is not analyzed in the baseline colonies dataset.||boolean|nuc-morph-analysis|
|termination|Manual annotation of 0 - track terminates by dividing. 1 - track terminates by going off the edge of the FOV. 2 - track terminates by apoptosis.|Trajectory termination annotation|NaN, 0, 1, or 2|nuc-morph-analysis|
|entering_mitosis|Is true if timepoint occurs in 2.5 hour window after lamin shell formation.||boolean|nuc-morph-analysis|
|exiting_mitosis|Is true if timepoint occurs in 30 minute window before lamin shell breakdown.||boolean|nuc-morph-analysis|
|entering_or_exiting_division|Is true if timepoint is flagged as entering mitosis or exiting mitosis. This filter is used to remove these timepoints when calculating the transient growth rate in the paper.||boolean|nuc-morph-analysis|
|neighbors|List of CellIds that are the adjacent nuclei, based on only nuclei centroids.||unitless|nuc-morph-analysis|
|neigh_distance|Mean distance from this nucleus's centroid to the centroids of neighboring nuclei.|Mean distance to neighbors|pixels|nuc-morph-analysis|
|is_tp_outlier|Is true if the timepoint is flagged as a single timepoint volume outlier. If true, the timepoint is not analyzed in the paper.||boolean|nuc-morph-analysis|
|track_length|The length of the nuclear trajectory in frames.||frames|nuc-morph-analysis|
|is_outlier_by_short_track|Is true if the track length has less than 5 timepoints. If true, the nuclear trajectory is not analyzed in the paper.||boolean|nuc-morph-analysis|
|is_outlier_curated_by_id|Is true if the nuclear trajectory is identified as a measurement outlier (i.e. segmentation issue, tracking issue, etc.).||boolean|nuc-morph-analysis|
|is_growth_outlier|Is true if the nuclear trajectory is identified as a biological outlier (i.e. grows for an abnormally long time and the daughters die).||boolean|nuc-morph-analysis|
|is_outlier_track|Is true for the combines flags: is_outlier_by_short_track, is_outlier_curated_by_id, and optionally is_growth_outlier.||boolean|nuc-morph-analysis|
|is_outlier|Is true for the combined flags: is_tp_outlier, is_after_breakdown_before_formation_outlier, is_outlier_track.|Growth Outlier Filter|boolean|nuc-morph-analysis|
|parent_id|The manually curated track_id of the parent cell. If the cell was manually identified to have no parent, this is -1. Otherwise NaN. This column is used to find mother-daughter and sister pairs in the paper.|Parent ID|unitless|nuc-morph-analysis|
|family_id|Unique identifier for all the nuclei in the same family tree.|Family ID|unitless|nuc-morph-analysis|
|distance_from_centroid|Distance from the centroid of the nuclear segmentation to the colony center.||microns|nuc-morph-analysis|
|colony_depth|Nuclei on the colony boundary have a depth of one. Nuclei are then assigned an increasing colony depth value using the Voronoi tessellation graph adjacency to all nuclei in the field of view (FOV).|Colony depth|unitless|nuc-morph-analysis|
|normalized_colony_depth|The normalized radial position in the colony, where the center of the colony is 0 and the edge is 1 calculated by (maximum depth of the colony - the nuclear depth) / (maximum depth of the colony - minimum depth of the colony) using the Voronoi-graph-based colony_depth feature.|Normalized distance from colony center|unitless|nuc-morph-analysis|
|normalized_distance_from_centroid|Distance of nucleus from centroid of colony / the maximum radius of the colony at each timepoint.||unitless|nuc-morph-analysis|
|colony_edge_in_fov|How much of the colony boundary is outside the FOV.||"full", "partial" or "none"|nuc-morph-analysis|
|colony_time|For the baseline colony dataset, the index sequence that aligns the Small, Medium, and Large colonies in their development.|Aligned colony time|frames|nuc-morph-analysis|
|non_interphase_volume|True if the volume is outside the distribution of volumes for full-interphase nuclear trajectories.||boolean|nuc-morph-analysis|
|non_interphase_mesh_sa|True if the surface area is outside the distribution of volumes for full-interphase nuclear trajectories.||boolean|nuc-morph-analysis|
|non_interphase_SA_vol_ratio|True if surface area to volume ratio are outside the distribution of volumes for full-interphase nuclear trajectories.||boolean|nuc-morph-analysis|
|non_interphase_size_shape|True if volumes are outside the distribution of volumes for full-interphase nuclear trajectories based on the volume, surface area and surface area to volume ratio. Used to detect interphase timepoints for transient growth rate measurements of nuclei that are not full-interphase trajectories.||boolean|nuc-morph-analysis|
|dxdt_48_volume|The transient growth rate. The change in volume over index sequence for a 4 hour rolling window for middle interphase timepoints of full interphase trajectories.|Transient growth rate|pixels cubed / frames|nuc-morph-analysis|
|neighbor_avg_volume_90um|The average volume of neighboring nuclei in the middle interphase nuclei in a 90 µm radius at each timepoint.||pixels cubed|nuc-morph-analysis|
|neighbor_avg_dxdt_48_volume_90um|The transient growth rate of neighboring nuclei in a 90 µm neighborhood. The average change in volume over index sequence for a 4 hour rolling window for middle interphase timepoints of nuclei in a 90 µm radius.|Average transient growth rate in 90¬µm neighborhood|pixels cubed / frames|nuc-morph-analysis|
|neighbor_avg_volume_whole_colony|The average volume for middle interphase nuclei in the whole colony at every timepoint.||pixels cubed|nuc-morph-analysis|
|neighbor_avg_dxdt_48_volume_whole_colony|The average transient growth rate in the whole colony.  The average change in volume over index sequence for a 4 hour rolling window for middle interphase timepoints of nuclei in the whole colony.|Colony average transient growth rate|pixels cubed / frames|nuc-morph-analysis|
|normalized_time|The normalized time is a measure of time for a full-interphase nuclear trajectory that ranges from 0 to 1, where 0 represents the time of nuclear lamin shell formation and 1 represents the time of nuclear lamin shell breakdown.|Normalized interphase time|unitless|nuc-morph-analysis|
|frame_transition|The calculated time of transition between expansion and growth phases in single full-interphase nuclear trajectory.|Start of growth (time)|frames|nuc-morph-analysis|
|sync_time_Ff|Time, synchronized to start at the time of lamin shell formation for a single full-interphase nuclear trajectory.|Synchronized time|frames|nuc-morph-analysis|
|volume_at_A|Volume at the time of lamin shell formation for a full-interphase nuclear trajectory.|Volume at formation|microns cubed|nuc-morph-analysis|
|location_x_at_A|X location at the time of lamin shell formation for a full-interphase nuclear trajectory.||pixels|nuc-morph-analysis|
|location_y_at_A|Y location at the time of lamin shell formation for a full-interphase nuclear trajectory.||pixels|nuc-morph-analysis|
|time_at_A|Time at lamin shell formation for a full-interphase nuclear trajectory.|Formation time|hours|nuc-morph-analysis|
|colony_time_at_A|Aligned colony time at lamin shell formation for a full-interphase nuclear trajectory.||hours|nuc-morph-analysis|
|volume_at_B|Volume at the time of transition, the start of the growth phase for a full-interphase nuclear trajectory.|Volume at start of growth|microns cubed|nuc-morph-analysis|
|location_x_at_B|X location at the time of transition, the start of the growth phase for a full-interphase nuclear trajectory.||pixels|nuc-morph-analysis|
|location_y_at_B|Y location at the time of  transition, the start of the growth phase for a full-interphase nuclear trajectory.||pixels|nuc-morph-analysis|
|time_at_B|Time at transition, the start of the growth phase for a full-interphase nuclear trajectory.||hours|nuc-morph-analysis|
|colony_time_at_B|Aligned colony time at transition, the start of the growth phase for a full-interphase nuclear trajectory.|Aligned colony time at start of growth|hours|nuc-morph-analysis|
|volume_at_C|Volume at the time of lamin shell breakdown for a full-interphase nuclear trajectory.|Volume at end of growth|microns cubed|nuc-morph-analysis|
|location_x_at_C|X location at the time of lamin shell breakdown for a full-interphase nuclear trajectory.||pixels|nuc-morph-analysis|
|location_y_at_C|Y location at the time of lamin shell breakdown for a full-interphase nuclear trajectory.||pixels|nuc-morph-analysis|
|time_at_C|Time at transition, the start of lamin shell breakdown for a full-interphase nuclear trajectory.|End of growth (time)|hours|nuc-morph-analysis|
|colony_time_at_C|Aligned colony time at lamin shell breakdown for a full-interphase nuclear trajectory.||hours|nuc-morph-analysis|
|duration_AB|Duration of the expansion phase from lamin shell formation to transition for a full-interphase nuclear trajectory.|Expansion duration|frames|nuc-morph-analysis|
|duration_BC|Duration of the growth phase from transition to lamin shell breakdown for a full-interphase nuclear trajectory.|Growth duration|frames|nuc-morph-analysis|
|duration_AC|The amount of volume added from transition to breakdown for a full-interphase nuclear trajectory.|Total interphase Duration|frames|nuc-morph-analysis|
|delta_volume_BC|The amount of volume added from transition to breakdown for a full-interphase nuclear trajectory.|Added volume during growth|microns cubed|nuc-morph-analysis|
|volume_fold_change_BC|The volume fold-change from transition to breakdown for a full-interphase nuclear trajectory.|Growth volume fold change|unitless|nuc-morph-analysis|
|SA_at_B|Surface area at the time of transition, the start of the growth phase for a full-interphase nuclear trajectory.|Surface area at start of growth|microns squared|nuc-morph-analysis|
|SA_at_C|Surface area at the time of lamin shell breakdown for a full-interphase nuclear trajectory.|Surface area at end of growth|microns squared|nuc-morph-analysis|
|delta_SA_BC|The amount of surface area added from transition to breakdown for a full-interphase nuclear trajectory.|Added surface area during growth|microns squared|nuc-morph-analysis|
|SA_fold_change_BC|The surface area fold change from transition to breakdown for a full-interphase nuclear trajectory.|Surface area fold change during growth|unitless|nuc-morph-analysis|
|volume_fold_change_fromB|The volume fold-change relative to the volume at transition, the start of the growth phase for a full-interphase nuclear trajectory.|Volume fold-change relative to starting volume|unitless|nuc-morph-analysis|
|SA_fold_change_fromB|The surface area fold-change relative to the volume at transition, the start of the growth phase for a full-interphase nuclear trajectory.|Surface area fold-change relative to starting surface area|unitless|nuc-morph-analysis|
|growth_rate_AB|The growth rate of the expansion phase calculated by endpoints: (volume_at_B - volume_at_A) / duration_AB for a full-interphase nuclear trajectory.|Expansion rate|microns cubed / hr|nuc-morph-analysis|
|late_growth_rate_by_endpoints|The growth rate of the growth phase calculated by endpoints: (volume_at_C - volume_at_B) / duration_BC for a full-interphase nuclear trajectory.|Growth rate|microns cubed / frames|nuc-morph-analysis|
|tscale_linearityfit_volume|Each volume trajectory from transition to breakdown was fit to a power law scaling with time 𝑉(𝑡)=𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼. This column is the fitted time scaling factor 𝛼 for a full-interphase nuclear trajectory.|Fitted time scaling factor|unitless|nuc-morph-analysis|
|atB_linearityfit_volume|Each volume trajectory from transition to breakdown was fit to a power law scaling with time 𝑉(𝑡)=𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼. This column is the starting volume fit parameter 𝑉𝑠𝑡𝑎𝑟𝑡 for a full-interphase nuclear trajectory.||microns cubed|nuc-morph-analysis|
|rate_linearityfit_volume|Each volume trajectory from transition to breakdown was fit to a power law scaling with time 𝑉(𝑡)=𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼. This column is the rate fit parameter r for a full-interphase nuclear trajectory.||1/hr|nuc-morph-analysis|
|RMSE_linearityfit_volume|Each volume trajectory from transition to breakdown was fit to a power law scaling with time 𝑉(𝑡)=𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼. This column is the root mean squared error of the fitted volume compared to the actual volume trajectory for a full-interphase nuclear trajectory.|Fitted volume root mean squared error (Power law)|microns cubed|nuc-morph-analysis|
|is_full_track|Is true if the timepoint is part of a full interphase trajectory for a full-interphase nuclear trajectory .||boolean|nuc-morph-analysis|
|exploratory_dataset|Baseline colonies column to filter datasets for visualization in timelapse feature explorer.|Exploratory dataset filter|boolean|nuc-morph-analysis|
|baseline_colonies_dataset|Baseline colonies column to filter datasets for visualization in timelapse feature explorer.|Baseline colonies dataset filter|boolean|nuc-morph-analysis|
|full_interphase_dataset|Baseline colonies column to filter datasets for visualization in timelapse feature explorer.|Full interphase analysis dataset filter|boolean|nuc-morph-analysis|
|lineage_annotated_dataset|Baseline colonies column to filter datasets for visualization in timelapse feature explorer.|Lineage annotated analysis dataset filter|boolean|nuc-morph-analysis|
|frame_of_formation|True if the the frame is the predicted lamin shell formation timepoint.||boolean|nuc-morph-analysis|
|frame_of_breakdown|True if the the frame is the predicted lamin shell breakdown timepoint.||boolean|nuc-morph-analysis|
|has_mitotic_neighbor_formation|True if the nucleus at a given timepoint has a directly adjacent neighbor undergoing lamin shell formation.||boolean|nuc-morph-analysis|
|has_mitotic_neighbor_breakdown|True if the nucleus at a given timepoint has a directly adjacent neighbor undergoing lamin shell breakdown.||boolean|nuc-morph-analysis|
|has_mitotic_neighbor|True if the nucleus at a given timepoint has a directly adjacent neighbor undergoing mitosis identified by a single lamin shell breakdown or lamin shell formation event.|Mitotic neighbor identified|boolean|nuc-morph-analysis|
|has_dying_neighbor|True if the nucleus at a given timepoint has a directly adjacent neighbor undergoing cell death.||boolean|nuc-morph-analysis|
|sum_has_dying_neighbor|Sum of dying adjacent neighbors over lifetime for a full-interphase nuclear trajectory.||events|nuc-morph-analysis|
|sum_has_mitotic_neighbor|Sum of mitotic adjacent neighbors over lifetime for a full-interphase nuclear trajectory.||events|nuc-morph-analysis|
|identified_death|Index sequence of cell death. Identified as the final frame when a nucleus has a termination value of 2.||frames|nuc-morph-analysis|
|frame_of_death|True if the the frame is the timepoint of cell death.||boolean|nuc-morph-analysis|
|has_dying_neighbor_forward_dilated|True if has_dying_neighbor is true, along with the 6 frames following the cell death timepoint.|Has dying neighbor flag|boolean|nuc-morph-analysis|
|has_mitotic_neighbor_breakdown_forward_dilated|If the nucleus has a directly adjacent neighbor that undergoes lamin shell breakdown, this feature is true at the timepoint of that lamin shell breakdown and at the six following timepoints.||boolean|nuc-morph-analysis|
|has_mitotic_neighbor_formation_backward_dilated|If the nucleus has a directly adjacent neighbor that undergoes lamin shell breakdown, this feature is true at the timepoint of that lamin shell breakdown and at the six prior timepoints.||boolean|nuc-morph-analysis|
|has_mitotic_neighbor_dilated|True if has_mitotic_neighbor_breakdown_forward_dilated or has_mitotic_neighbor_formation_backward_dilated is true, capturing all timepoints when neighbors are undergoing mitosis.|Has mitotic neighbor flag|boolean|nuc-morph-analysis|
|normalized_sum_has_mitotic_neighbor|Frequency of mitotic adjacent neighbors for a full-interphase nuclear trajectory. Calculated by the sum of mitotic adjacent nuclei normalized by the duration of the growth phase.|Frequency of mitotic adjacent neighbors|events/frames|nuc-morph-analysis|
|normalized_sum_has_dying_neighbor|Frequency of dying adjacent neighbors for a full-interphase nuclear trajectory. Calculated by the sum of dying adjacent nuclei normalized by the duration of the growth phase.|Frequency of dying adjacent neighbors|events/frames|nuc-morph-analysis|
|neighbor_avg_lrm_volume_90um|The average volume of neighboring nuclei in a 90 µm radius at each timepoint.||pixels cubed / frames|nuc-morph-analysis|
|neighbor_avg_lrm_height_90um|The average height of neighboring nuclei in a 90 µm radius at each timepoint.||pixels|nuc-morph-analysis|
|neighbor_avg_lrm_xy_aspect_90um|The average XY aspect ratio of neighboring nuclei in a 90 µm radius at each timepoint.||unitless|nuc-morph-analysis|
|neighbor_avg_lrm_mesh_sa_90um|The average surface area of neighboring nuclei in a 90 µm radius at each timepoint.||pixels squared|nuc-morph-analysis|
|neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um|The average density of neighboring nuclei in a 90 µm radius at each timepoint.||unitless|nuc-morph-analysis|
|mean_neighbor_avg_dxdt_48_volume_90um|The temporal mean of 'neighbor_avg_dxdt_48_volume_90um' over the lifetime of a full-interphase nuclear trajectory.|Neighborhood avg. mean transient growth rate|pixels cubed / frames|nuc-morph-analysis|
|mean_neighbor_avg_lrm_volume_90um|The temporal mean of 'neighbor_avg_lrm_volume_90um' over the lifetime of a full-interphase nuclear trajectory.|Neighborhood avg. mean volume|pixels cubed|nuc-morph-analysis|
|mean_neighbor_avg_lrm_height_90um|The temporal mean of 'neighbor_avg_lrm_height_90um' over the lifetime of a full-interphase nuclear trajectory.|Neighborhood avg. mean height|pixels|nuc-morph-analysis|
|mean_neighbor_avg_lrm_xy_aspect_90um|The temporal mean of 'neighbor_avg_lrm_xy_aspect_90um' over the lifetime of a full-interphase nuclear trajectory.|Neighborhood avg. mean XY aspect ratio|unitless|nuc-morph-analysis|
|mean_neighbor_avg_lrm_mesh_sa_90um|The temporal mean of 'neighbor_avg_lrm_mesh_sa_90um' over the lifetime of a full-interphase nuclear trajectory.|Neighborhood avg. mean surface area|pixels squared|nuc-morph-analysis|
|mean_neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um|The temporal mean of 'neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um' over the lifetime of a full-interphase nuclear trajectory.|Neighborhood avg. mean density|unitless|nuc-morph-analysis|
|early_neighbor_avg_dxdt_48_volume_90um|The average transient growth rate of neighboring nuclei within a 90 um radius around the start of the growth phase (2 hours after lamin shell formation) for a full-interphase nuclear trajectory.|Neighborhood avg. transient growth rate at ~start of growth|microns cubed / frames|nuc-morph-analysis|
|neighbor_avg_lrm_volume_90um_at_B|The average volume of neighboring nuclei within a 90 um radius at the start of the growth phase for a full-interphase nuclear trajectory.|Neighborhood avg. volume at start of growth|pixels cubed|nuc-morph-analysis|
|neighbor_avg_lrm_height_90um_at_B|The average height of neighboring nuclei within a 90 um radius at the start of the growth phase for a full-interphase nuclear trajectory.|Neighborhood avg. height at start of growth|pixels|nuc-morph-analysis|
|neighbor_avg_lrm_xy_aspect_90um_at_B|The average XY aspect ratio of neighboring nuclei within a 90 um radius at the start of the growth phase for a full-interphase nuclear trajectory.|Neighborhood avg. XY aspect ratio at start of growth|unitless|nuc-morph-analysis|
|neighbor_avg_lrm_mesh_sa_90um_at_B|The average surface area of neighboring nuclei within a 90 um radius at the start of the growth phase for a full-interphase nuclear trajectory.|Neighborhood avg. surface area at start of growth|pixels squared|nuc-morph-analysis|
|neighbor_avg_lrm_2d_area_nuc_cell_ratio_90um_at_B|The average density of neighboring nuclei within a 90 um radius at the start of the growth phase for a full-interphase nuclear trajectory.|Neighborhood avg. density at start of growth|unitless|nuc-morph-analysis|
|sisters_volume_at_B|A nucleus's sister's volume at start of growth for lineage annotated full-interphase nuclear trajectory.|Sisters starting volume|pixels cubed|nuc-morph-analysis|
|sisters_duration_BC|A nucleus's sister's duration of the growth phase  for lineage annotated full-interphase nuclear trajectory.|Sisters growth duration|pixels cubed / frames|nuc-morph-analysis|
|sisters_delta_volume_BC|A nucleus's sister's added volume for lineage annotated full-interphase nuclear trajectory.|Sisters added volume|pixels cubed|nuc-morph-analysis|
|height_at_B|The height at the start of the growth phase for a full-interphase nuclear trajectory.|Height at start of growth|pixels|nuc-morph-analysis|
|xy_aspect_at_B|The XY aspect ratio at the start of the growth phase for a full-interphase nuclear trajectory.|XY aspect ratio at start of growth|unitless|nuc-morph-analysis|
|SA_vol_ratio_at_B|The surface area to volume ratio at the start of the growth phase for a full-interphase nuclear trajectory.|Surface area/volume ratio at start of growth|pixels squared|nuc-morph-analysis|
|2d_area_nucleus|The area of the nucleus (from maximum z projected nuclear segmentation).|2D nuclear area|pixels squared|nuc-morph-analysis|
|2d_area_pseudo_cell|The area of the pseudo cell (from maximum z projected nuclear segmentation).|Pseudo cell area|pixels squared|nuc-morph-analysis|
|2d_area_nuc_cell_ratio|The ratio between the '2d_area_nucleus' and the '2d_area_psuedo_cell' gives a metric represtative of the local density for each nucleus at every timepoint. If the density could not be properly calculated at this timepoint (ie. edge of colony, has a mitotic neighbor etc) the column value is NaN.|Density|unitless|nuc-morph-analysis|
|2d_perimeter_nucleus|The perimeter of the nucleus (from maximum z projected nuclear segmentation).||pixels|nuc-morph-analysis|
|2d_perimeter_pseudo_cell|The perimeter of the pseudo cell (from maximum z projected nuclear segmentation).||pixels|nuc-morph-analysis|
|2d_perimeter_nuc_cell_ratio|The ratio between the '2d_perimeter_nucleus' and the '2d_perimeter_pseudo_cell'.||unitless|nuc-morph-analysis|
|bad_pseudo_cells_segmentation|True if a pseudo cell segmentation correpsonding to the nucleus is known, a priori, to be erroneously large due to being at edge of colony or having neighbors that lack a segmentation, such as mitotic cells. (See supp Fig. S4D)||boolean|nuc-morph-analysis|
|uncaught_pseudo_cell_artifact|True if pseudo cell segmentation corresponding to the nucleus is abnormally large and not caught by the bad_pseudo_cells_segmentation filter. The criteria are: '2d_perimeter_nuc_cell_ratio' < 0.4 OR '2d_perimeter_pseudo_cell' > 500 pixels OR '2d_area_nuc_cell_ratio' < 0.2||boolean|nuc-morph-analysis|
|volume_change_over_25_minutes|The nuclear volume change in a 25 minute window at each frame (t) calculated by ∆V(t) = V(t-5) - V(t).|Change in volume in 25 minute window|pixels cubed|nuc-morph-analysis|
|power_fit_volume|Each volume trajectory from transition to breakdown was fit to a power law scaling with time 𝑉(𝑡)=𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼. This column is the power fit volume with timepoints from identified volume dips removed.|Power law fitted volume|voxels cubed|nuc-morph-analysis|
|volume_dips_peak_mask_at_region|True at all timepoints along  an identified volume dip event.|Volume dip flag|boolean|nuc-morph-analysis|
|volume_dips_peak_mask_at_center|True at the timepoint that is the "peak" (or minima) of an identified volume dip event.||boolean|nuc-morph-analysis|
|volume_dips_volume_change_at_center|The change in volume from the start of a volume dip to the "peak" (or minima) of a volume dip. Value only reported at the timepoint of the volume dip "peak" (or minima)||microns cubed|nuc-morph-analysis|
|volume_dips_removed_um_unfilled|The volume trajectory of a nucleus with the timepoints within each volume dip event region replaced with NaN.||microns cubed|nuc-morph-analysis|
|dxdt_48_volume_dips_removed_um_unfilled|The transient growth rate computed with volume dips removed (generated by computing transient growth rate using the `volume_dips_removed_um_unfilled' column. )||pixels cubed / frames|nuc-morph-analysis|
|neighbor_avg_dxdt_48_volume_dips_removed_um_unfilled_90um|The transient growth rate of neighboring nuclei in a 90 µm neighborhood with volume dips removed. The average change in volume over index sequence for a 4 hour rolling window for middle interphase timepoints of nuclei in a 90 µm radius. Volume measurements identified as volume dips were excluded.||pixels cubed / frames|nuc-morph-analysis|
|neighbor_avg_dxdt_48_volume_dips_removed_um_unfilled_whole_colony|The transient growth rate of all nuclei in the colony with volume dips removed. The average change in volume over index sequence for a 4 hour rolling window for middle interphase timepoints of nuclei in the whole colony. Volume measurements identified as volume dips were excluded.||pixels cubed / frames|nuc-morph-analysis|
|RMSE_exponentialfit_volume|Each volume trajectory from transition to breakdown was fit to an exponential scaling with time 𝑉(𝑡)=𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼. This column is the root mean squared error of the fitted volume compared to the actual volume trajectory for a full-interphase nuclear trajectory.||microns cubed|nuc-morph-analysis|
|RMSE_linearfit_volume|Each volume trajectory from transition to breakdown was fit to a linear scaling with time  𝑉(𝑡)=𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡. This column is the root mean squared error of the fitted volume compared to the actual volume trajectory for a full-interphase nuclear trajectory.||microns cubed|nuc-morph-analysis|