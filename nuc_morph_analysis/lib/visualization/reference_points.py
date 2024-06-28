COLONY_COLORS = {
    "small": "#1B9E77",
    "medium": "#D95F02",
    "large": "#7570B3",
    "baby_bear": "#1B9E77",
    "goldilocks": "#D95F02",
    "mama_bear": "#7570B3",
    "feeding_control_baseline": "#E7298A",
    "feeding_control_refeed": "#E6AB02",
    "feeding_control_starved": "#66A61E",
}

COLONY_LABELS = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
    "baby_bear": "Small",
    "goldilocks": "Medium",
    "mama_bear": "Large",
    "feeding_control_baseline": "Feeding control baseline",
    "feeding_control_starved": "Feeding control starved",
    "feeding_control_refeed": "Feeding control refeed",
}

OBSERVED_TOUCH_COLONY = {
    "small": 178,  # time index when colony touches another colony
    "medium": None,
    "large": None,
}

"""
To try to asses the effect of touching another colony, adding a reference point
for the time point when the two colonies have started to merge after touching. Merging 
is arbitrarily defined as when ~ 1/4 the of the colony edge is now merged with the other colony.
"""
OBSERVED_MERGE_COLONY = {
    "small": 236,  # time index when it looks like the colony is merging
    "medium": None,
    "large": None,
}

FOV_TOUCH_T_INDEX = {  # time index when colony edge first touches frame boundary
    "small": 180,
    "medium": 144,
    "large": 0,
}

FOV_EXIT_T_INDEX = {  # time index when colony edge is no longer in frame
    "small": 480,
    "medium": 384,
    "large": 216,
}
