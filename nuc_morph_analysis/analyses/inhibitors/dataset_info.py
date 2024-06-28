import numpy as np
import pandas as pd
from pathlib import Path
from collections import namedtuple
import re
from nuc_morph_analysis.lib.preprocessing import system_info
from typing import Dict, List

EXCLUDE_tracks_expansion: Dict[str, Dict[str, List[int]]] = {
    "aphidicolin_lamin_exp1_rep2": {
        "control": [],
        "perturb": [],
    },
    "aphidicolin_lamin_exp2_rep1": {
        "control": [],
        "perturb": [],
    },
    "importazole_lamin_exp1_rep1": {
        "control": [
            143545,  # no volume increase corresponding to expansion observed
        ],
        "perturb": [
            155005,  # exhibits sudden shift in volume that could be track switch
            155881,  # track irregularities in second half
        ],
    },
}


def drug_names_for_titles(input_str):
    drug_names = {
        "aphidicolin_lamin_exp1_rep2": "DNA replication (set#1, rep.#2)",
        "aphidicolin_lamin_exp1_controlsONLY": "DNA replication (set#1, controls only)",
        "aphidicolin_lamin_exp2_rep1": "DNA replication (set#2, rep.#1)",
        "importazole_lamin_exp1_rep1": "Nuclear import (set#1, rep.#1)",
    }
    return drug_names[input_str]


def drug_analysis_pairs():
    DrugPair = namedtuple("DrugPair", ["control", "perturb"])
    pair_dict = {
        "aphidicolin_lamin_exp1_rep2": DrugPair(
            "drug_perturbation_1_scene2", "drug_perturbation_1_scene4"
        ),
        "aphidicolin_lamin_exp1_controlsONLY": DrugPair(
            "drug_perturbation_1_scene0", "drug_perturbation_1_scene2"
        ),
        "aphidicolin_lamin_exp2_rep1": DrugPair(
            "drug_perturbation_4_scene2", "drug_perturbation_4_scene4"
        ),
        "importazole_lamin_exp1_rep1": DrugPair(
            "drug_perturbation_2_scene0", "drug_perturbation_2_scene6"
        ),
    }
    return pair_dict


def get_drug_perturbation_experiment_class_from_colony_name(colony):
    """
    return all information about a drug perturbation experiment for a given colony

    Parameters
    ----------
    colony : str
        name of the colony

    Returns
    -------
    dpclass : class
        class object containing all information about the drug perturbation experiment
    """
    dpname = re.search("drug_perturbation_[0-9]", colony).group(0)
    class_dict = get_dataset_as_class()
    dpclass = class_dict[dpname]
    return dpclass


def get_drug_perturbation_details(dpclass, scene_number, minimal=False):
    """
    return all information about a drug perturbation experiment for a given colony

    Parameters
    ----------
    dpclass : class
        class object containing all information about the drug perturbation experiment
    scene_number : str
        scene number

    Returns
    -------
    info : dict
        dictionary containing all information about the drug perturbation experiment
    """
    if minimal:
        info = {}
    else:
        info = {
            "num_dying_x": dpclass.dying_per_scene[scene_number].num_dying_x,
            "num_dying_y": dpclass.dying_per_scene[scene_number].num_dying_y,
        }
    update_dict = {
        "drug_added_frame": dpclass.drug_time,
        "time_interval_minutes": 5,
        "drugs_string": dpclass.drugs_per_scene[scene_number],
        "egfp_channel_index": dpclass.EGFP_channel_index,
        "pixel_size": system_info.PIXEL_SIZE_YX_100x,
    }
    info.update(update_dict)
    return info


def get_drug_perturbation_details_from_colony_name(colony):
    """
    return all information about a drug perturbation experiment for a given colony

    Parameters
    ----------
    colony : str
        name of the colony

    Returns
    -------
    info : dict
        dictionary containing all information about the drug perturbation experiment
    """
    dpclass = get_drug_perturbation_experiment_class_from_colony_name(colony)
    scene = re.search("scene[0-9]+", colony).group(0)
    scene_number = re.search("[0-9]+", scene).group(0)
    info = get_drug_perturbation_details(dpclass, scene_number)
    return info


def preprocess_and_add_columns(dfo, dict_of_control_perturb_colonies):
    """
    this function slices the dataframe to only include the control and perturb colonies,
    and adds columns for condition, volume_um, time_minutes, and time_minutes_since_drug

    Parameters
    ----------
    dfo : pandas.DataFrame
        dataframe containing all colonies
    dict_of_control_perturb_colonies : dict
        dictionary containing the control and perturb colony names

    Returns
    -------
    df : pandas.DataFrame
        dataframe containing only the control and perturb colonies with added columns
    """
    df = dfo[dfo.colony.isin(list(dict_of_control_perturb_colonies.values()))].copy()
    for control_perturb, colony in dict_of_control_perturb_colonies.items():
        details = get_drug_perturbation_details_from_colony_name(colony)

        df.loc[df["colony"] == colony, "condition"] = control_perturb
        idx = df["colony"] == colony
        df.loc[idx, "volume_um"] = df.loc[idx, "volume"] * details["pixel_size"] ** 3
    df["time_minutes"] = df["index_sequence"] * details["time_interval_minutes"]
    df["time_minutes_since_drug"] = (
        df["time_minutes"] - details["drug_added_frame"] * details["time_interval_minutes"]
    )
    return df


# define a named tuple to store the dying arrays to record percentage of colony dead (num_dying_y) at a given time (num_dying_x)
DyingArrays = namedtuple("DyingArrays", ["num_dying_x", "num_dying_y"])


def get_dataset_as_class():
    class DrugPerturbation:
        def __init__(
            self,
            name,
            instrument,
            fms_id,
            drug_time,
            barcode,
            drugs_per_well,
            drugs_per_scene,
            colony_names_per_scene,
            names_per_scene,
            dying_per_scene,
            EGFP_channel_index,
        ):
            self.name = name
            self.instrument = instrument
            self.fms_id = fms_id
            self.drug_time = drug_time
            self.barcode = barcode
            self.drugs_per_well = drugs_per_well
            self.drugs_per_scene = drugs_per_scene
            self.colony_names_per_scene = colony_names_per_scene
            self.names_per_scene = names_per_scene
            self.dying_per_scene = dying_per_scene
            self.EGFP_channel_index = EGFP_channel_index

    # Create instances of the class
    drug_perturbation1 = DrugPerturbation(
        name="20230424",
        instrument="ZSD-3",
        fms_id="5d1d2b2d6e0b40b4968e92f5113ec498",
        drug_time=45,  # zen= 45^46, python=44^45
        barcode="AD00004205",
        drugs_per_well={
            "B5": "control (media only)",
            "B6": "aphidicolin (20.2 ug/mL)",
            "B7": "aphidicolin (4.8 ug/mL)",
            "B8": "control (nothing)",
        },
        drugs_per_scene={
            "0": "control (media only)",
            "1": "control (media only)",
            "2": "control (media only)",
            "3": "aphidicolin (20.2 ug/mL)",
            "4": "aphidicolin (20.2 ug/mL)",
            "5": "aphidicolin (20.2 ug/mL)",
            "6": "aphidicolin (4.8 ug/mL)",
            "7": "aphidicolin (4.8 ug/mL)",
            "8": "aphidicolin (4.8 ug/mL)",
            "9": "aphidicolin (4.8 ug/mL)",
            "10": "aphidicolin (4.8 ug/mL)",
            "11": "control (nothing)",
            "12": "control (nothing)",
            "13": "control (nothing)",
        },
        colony_names_per_scene={
            "0": "drug_perturbation_1_scene0",
            "2": "drug_perturbation_1_scene2",
            "4": "drug_perturbation_1_scene4",
        },
        names_per_scene={
            "0": "control1",
            "2": "control2",
            "4": "perturb2",
        },
        # fmt: off
        dying_per_scene={
            '0' : DyingArrays(np.asarray([1, 65, 74, 80, 87, 95,])-1, np.asarray([0, 0, 0, 0, 0, 0,])),
            '2' : DyingArrays(np.asarray([1, 65, 74, 80, 87, 95,])-1, np.asarray([0, 0, 0, 0, 0, 0,])),
            '4' : DyingArrays(np.asarray([1, 65, 74, 80, 87, 95,])-1, np.asarray([0, 2, 4, 10, 20, 50,])),
        },
        # fmt: on
        EGFP_channel_index=1,
    )

    drug_perturbation2 = DrugPerturbation(
        name="20230417",
        instrument="ZSD-3",
        fms_id="f648379a70b049dcb5cbfc1893411a97",
        drug_time=36,  # zen= 36^37, python=35^36
        barcode="AD00003974",
        drugs_per_well={
            "C5": "control (mixing only)",
            "D5": "2-aminopurine (20.2 ng/mL)",
            "D6": "importazole (127.3 uM)",
            "D7": "importazole (47.2 uM)",
            "D8": "puromycin (2 ug/mL)",
        },
        drugs_per_scene={
            "0": "control (mixing only)",
            "1": "control (mixing only)",
            "2": "control (mixing only)",
            "3": "2-aminopurine (20.2 ng/mL)",
            "4": "2-aminopurine (20.2 ng/mL)",
            "5": "2-aminopurine (20.2 ng/mL)",
            "6": "importazole (127.3 uM)",
            "7": "importazole (127.3 uM)",
            "8": "importazole (127.3 uM)",
            "9": "importazole (47.2 uM)",
            "10": "importazole (47.2 uM)",
            "11": "importazole (47.2 uM)",
            "12": "puromycin (2 ug/mL)",
            "13": "puromycin (2 ug/mL)",
            "14": "puromycin (2 ug/mL)",
        },
        colony_names_per_scene={
            "0": "drug_perturbation_2_scene0",
            "6": "drug_perturbation_2_scene6",
        },
        names_per_scene={
            "0": "control1",
            "6": "perturb2",
        },
        # fmt: off
        dying_per_scene={
            '0' : DyingArrays(np.asarray([1, 45, 47, 63, 72, 84, 90, 107, 139,])-1, np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0,])),
            '6' : DyingArrays(np.asarray([1, 47, 53, 63,  69, 75, 85, ])-1, np.asarray([0, 0 , 2 ,  10 , 35, 50, 85, ])),
        },
        # fmt: on
        EGFP_channel_index=1,
    )

    drug_perturbation4 = DrugPerturbation(
        name="20220411",
        instrument="ZSD-2",
        fms_id="1d284940dc8f4b3d8360f85b5b9beee7",
        drug_time=38,  # zen= 38^39, python=37^38
        barcode="AD00000216",
        drugs_per_well={
            "E4": "control (media only)",
            "E5": "aphidicolin (20.2 ug/mL)",
            "E6": "aphidicolin (11.2 ug/mL)",
            "E7": "aphidicolin (5.9 ug/mL)",
            "E8": "rapamycin (9.1 uM)",
        },
        drugs_per_scene={
            "0": "control (media only)",
            "1": "control (media only)",
            "2": "control (media only)",
            "3": "aphidicolin (20.2 ug/mL)",
            "4": "aphidicolin (20.2 ug/mL)",
            "5": "aphidicolin (20.2 ug/mL)",
            "6": "aphidicolin (11.2 ug/mL)",
            "7": "aphidicolin (11.2 ug/mL)",
            "8": "aphidicolin (11.2 ug/mL)",
            "9": "aphidicolin (5.9 ug/mL)",
            "10": "aphidicolin (5.9 ug/mL)",
            "11": "aphidicolin (5.9 ug/mL)",
            "12": "rapamycin (9.1 uM)",
            "13": "rapamycin (9.1 uM)",
            "14": "rapamycin (9.1 uM)",
        },
        colony_names_per_scene={
            "2": "drug_perturbation_4_scene2",
            "4": "drug_perturbation_4_scene4",
        },
        names_per_scene={
            "2": "control1",
            "4": "perturb1",
        },
        # fmt: off
        dying_per_scene={
            '2' : DyingArrays(np.asarray([1, 68, 69, 72, 77, 81, 87])-1, np.asarray([0, 0, 0, 0, 0, 0,])),
            '4' : DyingArrays(np.asarray([1, 68, 69, 72, 77, 81, 87])-1, np.asarray([0, 0, 1, 4, 8, 15, 50,])),
        },
        # fmt: on
        EGFP_channel_index=1,
    )

    class_dict = {
        "drug_perturbation_1": drug_perturbation1,
        "drug_perturbation_2": drug_perturbation2,
        "drug_perturbation_4": drug_perturbation4,
    }
    return class_dict
