# Load data

Loading for baseline or control colony datasets in a uniform way and calculates various features for each nucleus at different timepoints. This dataset contains all timepoints. 

```
from nuc_morph_analysis.lib.preprocessing import global_dataset_filtering
df = global_dataset_filtering.load_dataset_with_features() # default is "all_baseline"
df_medium = global_dataset_filtering.load_dataset_with_features('medium')
```

```
df = global_dataset_filtering.load_dataset_with_features("all_feeding_control")
df_control = global_dataset_filtering.load_dataset_with_features("feeding_control_baseline")
```

# Filter data

To apply minimal filtering of apoptotic nuclei, single-timepoint outliers, 
and nuclei cut off by the fov edge to get a dataset with all tracks all timepoints:

`df_filt = filter_data.all_timepoints_minimal_filtering(df)`

To get a dataset of full tracks containing all timepoints, with the above minimal filters applied:

`df_full = filter_data.all_timepoints_full_tracks(df)`

To get a dataset of full tracks containing a single row per track:

`track_level_feature_df = filter_data.track_level_features(df)`

## Adding precalculated features

When you develop a new feature to be added to an analysis, please add it to the `global_dataset_filtering.py` so that others can
use the same feature across workflows. 

If it is an all timepoint feature it can be added to: `global_dataset_filtering.process_all_tracks()`
If it is a full_track feature it can be added to: `global_dataset_filtering.process_full_tracks()`

The feature can be calculated in your analysis folder or added to `nuc_morph_analysis.lib.preprocessing.add_features.py`

If the feature takes a lot of time or needs to be done on all datasets at once you can consider adding it to the `generate_main_manifest` workflow.
