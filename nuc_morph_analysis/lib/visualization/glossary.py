GLOSSARY = {
    "volume": "The sum of the voxels inside the nuclear segmentation at every time point.",
    "height": "The distance between the lowest and highest pixels of the nucleus segmentation in the Z-dimension at every time point.",
    "xy_aspect": "The ratio of length to the width of the nuclear segmentation at every timepoint. The length is defined as the longest axis of the nuclear segmentation in the XY-plane. The width is defined as the length of nuclear segmentation in the plane perpendicular to the longest axis.",
    "family_id": "Unique identifier for all the nuclei in the same family tree.",
    "volume_at_B": "The volume at the start of the growth phase for a full-interphase nuclear trajectory.",
    "volume_at_C": "The volume at the time of lamin shell breakdown, the end of the growth phase, for a full-interphase nuclear trajectory.",
    "volume_fold_change_BC": "The volume fold-change from the start to the end of growth for a full-interphase nuclear trajectory (i.e., “volume at the end of growth” / “volume at start of growth”).",
    "delta_volume_BC": "The amount of volume added from the start to the end of growth for a full-interphase nuclear trajectory (i.e., “volume at the end of growth” - “volume at start of growth”).",
    "duration_BC": "Duration of the growth phase from the start to the end of growth for a full-interphase nuclear trajectory.",
    "late_growth_rate_by_endpoints": "The growth rate of the growth phase calculated by endpoints (i.e., “volume at the end of growth” - “volume at start of growth”) / “growth duration”.",
    "tscale_linearityfit_volume": "Each volume trajectory was fit to a power law scaling with time 𝑉(𝑡) =𝑉𝑠𝑡𝑎𝑟𝑡+𝑟𝑡^𝛼 over all time points during the growth phase. This feature is the fitted time scaling factor 𝛼 for a full-interphase nuclear trajectory.",
    "dxdt_48_volume": "The change in volume over time for a four hour rolling window for middle interphase time points of full-interphase trajectories.",
}
# Colored segmentation: The calculated feature is available for that nucleus.
# Grey segmentation: The calculated feature is not available for that nucleus. This could be because the nuclear segmentation is an outlier at that time point (i.e. touching the edge of the field of view, identified as an erroneous segmentation or tracking) or the feature could not be calculated (i.e. features that require the full-interphase trajectory).
