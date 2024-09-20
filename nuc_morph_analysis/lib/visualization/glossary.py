GLOSSARY = {
    "volume": "The sum of the voxels inside the nuclear segmentation at every timepoint.",
    "height": "The distance between the lowest and highest pixels of the nucleus segmentation in the Z-dimension at every timepoint.",
    "xy_aspect": "The ratio of length to the width of the nuclear segmentation at every timepoint. The length is defined as the longest axis of the nuclear segmentation in the XY-plane. The width is defined as the length of nuclear segmentation in the plane perpendicular to the longest axis.",
    "family_id": "Unique identifier for all the nuclei in the same family tree.",
    "volume_at_B": "The volume at the start of the growth phase for a full-interphase nuclear trajectory.",
    "volume_at_C": "The volume at the time of lamin shell breakdown, the end of the growth phase, for a full-interphase nuclear trajectory.",
    "volume_fold_change_BC": "The volume fold-change from the start to the end of growth for a full-interphase nuclear trajectory (i.e., “volume at the end of growth” / “volume at start of growth”).",
    "delta_volume_BC": "The amount of volume added from the start to the end of growth for a full-interphase nuclear trajectory (i.e., “volume at the end of growth” - “volume at start of growth”).",
    "duration_BC": "Duration of the growth phase from the start to the end of growth for a full-interphase nuclear trajectory.",
    "late_growth_rate_by_endpoints": "The growth rate of the growth phase calculated by endpoints (i.e., “volume at the end of growth” - “volume at start of growth”) / “growth duration”.",
    "tscale_linearityfit_volume": "Each volume trajectory was fit to a power law scaling with time 𝑉(𝑡) =𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼 over all timepoints during the growth phase. This feature is the fitted time scaling factor 𝛼 for a full-interphase nuclear trajectory.",
    "dxdt_48_volume": "The change in volume over time for a four hour rolling window for middle interphase timepoints of full-interphase trajectories.",
    "density": "Inverse of the squared average distance to centroids of all neighboring nuclei. Neighbors were determined using a Voronoi tesselation graph.",
    "normalized_time": "The time within interphase normalized by the total interphase time for full-interphase nuclear trajectories. This ranges from 0 to 1, where 0 represents the start of interphase and 1 represents the end of interphase.",
    "sync_time_Ff": "Time synchronized to start of interphase for each single full-interphase nuclear trajectory (i.e., all trajectories start with a synchronized time of 0 hours).",
    "time_at_B": "The calculated time of the start of the growth phase in single full-interphase nuclear trajectory. This time is relative to the start of the timelapse imaging.",
    "colony_time_at_B": "In general, “aligned colony time” is the universal timeline for all three baseline colonies (Small, Medium and Large), based on aligning their individual timelapses based on their colony development. This feature gives the time for the start of the growth phase for an individual full-interphase nuclear trajectory within aligned colony time.",
    "normalized_colony_depth": "The normalized radial position in the colony, where the center of the colony is 0 and the edge is 1. The colony depth of each nucleus is assigned using a Voronoi tesselation graph. The normalized distance from the center is calculated from these depths as (maximum depth within the colony - individual nuclear depth) / (maximum depth within the colony - minimum depth within the colony)",
    "termination": "Manual annotation of how a trajectory terminated. 0 - trajectory terminates by cell dividing. 1 - trajectory terminates by nucleus going off the edge of the FOV. 2 - trajectory terminates by apoptosis.",
    "is_growth_outlier": "Is true if the nuclear trajectory is identified as a biological outlier (e.g. grows for an abnormally long time and the daughters die)",
    "baseline_colonies_dataset": "Filter which limits the included dataset to nuclei in the “baseline colonies analysis dataset.” This includes all nuclei tracked for at least one hour and included “growth outliers” (biological outliers) but excludes any technical outliers (outliers automatically filtered or annotated as having errors in segmentation, tracking, etc).",
    "full_interphase_dataset": "Filter which limits the included dataset to nuclei analyzed in the “full-interphase analysis dataset.” It is a subset of the “baseline colonies analysis dataset.” Only nuclei tracked successfully throughout interphase are included in this dataset, and growth outliers are excluded from this dataset.",
    "lineage_annotated_dataset": "Filter which limits the included dataset to nuclei analyzed in the “lineage-annotated analysis dataset.” It is a subset of the “full-interphase dataset,” including just the nuclei in the Small and Medium colonies from this dataset. It includes the “Family ID” feature.",
    "volume_at_A": "The volume at the time of lamin shell formation, the start of the expansion phase, for a full-interphase nuclear trajectory.",
    "time_at_A": "Time at lamin shell formation, the start of the expansion phase, for a full-interphase nuclear trajectory.",
    "time_at_C": "Time at lamin shell breakdown, the end of the growth phase, for a full-interphase nuclear trajectory.",
    "duration_AB": "Duration of the expansion phase from lamin shell formation to the start of the growth phase for a full-interphase nuclear trajectory.",
    "duration_AC": "Duration of the total time during interphase from lamin shell formation (which is the start of the expansion phase) to lamin shell breakdown (which is the end of the growth phase) for a full-interphase nuclear trajectory. Interphase includes both the “expansion” phase and the “growth” phase.",
    "growth_rate_AB": "The growth rate of the expansion phase calculated by endpoints: (volume at end of expansion - volume at start of expansion) / expansion duration for a full-interphase nuclear trajectory.",
    "volume_fold_change_fromB": "The volume fold-change relative to the volume at the start of growth for a full-interphase nuclear trajectory (i.e. volume / volume at start of growth).",
    "distance_from_centroid": "Distance from the center of the colony.",
    "neighbor_avg_dxdt_48_volume_whole_colony": "The transient growth rate over a four hour rolling window centered at a given timepoint, averaged across all nuclei in the colony.",
    "neighbor_avg_dxdt_48_volume_90um": "The transient growth rate over a four hour rolling window centered at a given timepoint, averaged across all nuclei within a 90 µm radius neighborhood of each nucleus.",
    "zy_aspect": "The ratio of width to the height of the nuclear segmentation at every timepoint. The width is defined as the length of nuclear segmentation in the plane perpendicular to the longest axis in the XY-plane. The height is the length of the nuclear segmentation in the Z-plane.",
    "xz_aspect": "The ratio of width to the height of the nuclear segmentation at every timepoint. The length is defined as the longest axis of the nuclear segmentation in the XY-plane. The height is the length of the nuclear segmentation in the Z-plane.",
    "mesh_sa": "The number of pixels on the surface of the smoothed mesh of the nuclear segmentation at every timepoint.",
    "SA_at_B": "The surface area at the start of the growth phase for a full-interphase nuclear trajectory.",
    "SA_at_C": "The surface area at the time of lamin shell breakdown, the end of the growth phase, for a full-interphase nuclear trajectory.",
    "SA_fold_change_BC": "The surface area fold-change from the start to the end of growth for a full-interphase nuclear trajectory (i.e., “surface area at the end of growth” / “surface area at start of growth”).",
    "SA_fold_change_fromB": "The surface area fold-change relative to the surface area at the start of growth for a full-interphase nuclear trajectory (i.e. surface area / surface area at start of growth).",
    "delta_SA_BC": "The amount of surface area added from the start to the end of growth for a full-interphase nuclear trajectory (i.e., “surface area at the end of growth” - “surface area at start of growth”).",
    "SA_vol_ratio": "The ratio of the surface area to the volume of the nuclear segmentation at every timepoint.",
}
# Colored segmentation: The calculated feature is available for that nucleus.
# Grey segmentation: The calculated feature is not available for that nucleus. This could be because the nuclear segmentation is an outlier at that timepoint (i.e. touching the edge of the fieldof view (FOV), identified as an erroneous segmentation or tracking) or the feature could not be calculated (i.e. features that require the full-interphase trajectory).
