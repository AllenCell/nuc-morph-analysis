AXIAL_DISTORTION_CORRECTION_FACTOR_20x = 1.43  # empirically determined axial distortion correction factor. see Diel et al 2020 for discussion of axial distortion due to refractive index mismatch
AXIAL_DISTORTION_CORRECTION_FACTOR_100x = 1.00  # no axial distortion correction needed for 100x objective because refractive index is matched
CAMERA_PIXEL_SIZE = 6.5  # units of um
BINNING_20x = 1
BINNING_100x = 2
TOTAL_MAG_20x = 24  # 20x objective * 1.2 x camera relay lens
TOTAL_MAG_100x = 120  # 100x objective * 1.2 x camera relay lens
Z_STEP_SIZE_NOMINAL_20x = 0.53  # the nominal Z-step size for the 20x objective. The actual Z-step size is 0.53 * the axial distortion correction factor
Z_STEP_SIZE_NOMINAL_100x = 0.29  # the nominal (and actual) Z-step size for the 100x objective
Z_STEP_SIZE_ACTUAL_20x = Z_STEP_SIZE_NOMINAL_20x * AXIAL_DISTORTION_CORRECTION_FACTOR_20x
Z_STEP_SIZE_ACTUAL_100x = Z_STEP_SIZE_NOMINAL_100x * AXIAL_DISTORTION_CORRECTION_FACTOR_100x

# it's important to compute this without using pixel sizes because a float precision error will occur
# this value should be 0.4
RESCALE_FACTOR_100x_to_20X = (TOTAL_MAG_20x / BINNING_20x) / (TOTAL_MAG_100x / BINNING_100x)

# single pixel size for YX plane (pixel size for Y = pixel size for X)
PIXEL_SIZE_YX_100x = (CAMERA_PIXEL_SIZE * BINNING_100x) / TOTAL_MAG_100x
PIXEL_SIZE_YX_20x = (CAMERA_PIXEL_SIZE * BINNING_20x) / TOTAL_MAG_20x

# single pixel size for Z axis
PIXEL_SIZE_Z_100x = Z_STEP_SIZE_ACTUAL_100x
PIXEL_SIZE_Z_20x = Z_STEP_SIZE_ACTUAL_20x
