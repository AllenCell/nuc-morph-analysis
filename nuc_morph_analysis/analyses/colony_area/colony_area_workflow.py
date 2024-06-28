from pathlib import Path
from nuc_morph_analysis.lib.preprocessing import load_data
from nuc_morph_analysis.lib.preprocessing.system_info import PIXEL_SIZE_YX_20x
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import twoD_zMIP_helper


def _write_starting_area(dataset, area):
    # Brightfield images for baseline colonies are at 20x
    area_um2 = f"{area * (PIXEL_SIZE_YX_20x ** 2)} μm²"
    print(f"{dataset} area at frame 0: {area_um2}")

    path = Path(__file__).parent / "figures" / f"starting_area_{dataset}.txt"
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        f.write(f"{area} px\n")
        f.write(area_um2)
        f.write("\n")


def _starting_area(dataset):
    # Load the first frame
    fov_brightfield = load_data.get_raw_fov_at_timepoint(dataset, 0)

    # now segment the colony
    feats = twoD_zMIP_helper.segment_colony_brightfield(fov_brightfield)
    # save the starting area
    _write_starting_area(dataset, feats["colony_area"])


def baseline_colony_starting_area():
    """
    Segment each baseline colony at frame 0 and compute the area of the segmentation.
    """
    for dataset in ["small", "medium", "large"]:
        _starting_area(dataset)


def feeding_control_starting_area():
    """
    Segment each feeding control colony at frame 0 and compute the area of the segmentation.
    """
    for dataset in [
        "feeding_control_baseline",
        "feeding_control_starved",
        "feeding_control_refeed",
    ]:
        _starting_area(dataset)


def aphidicolin_control_starting_area():
    """
    Segment each feeding control colony at frame 0 and compute the area of the segmentation.
    """
    for dataset in [
        "drug_perturbation_1_scene0",
        "drug_perturbation_1_scene2",
        "drug_perturbation_1_scene4",
        "drug_perturbation_4_scene2",
        "drug_perturbation_4_scene4",
    ]:
        _starting_area(dataset)


def importazole_control_starting_area():
    """
    Segment each feeding control colony at frame 0 and compute the area of the segmentation.
    """
    for dataset in ["drug_perturbation_2_scene6", "drug_perturbation_2_scene0"]:
        _starting_area(dataset)
