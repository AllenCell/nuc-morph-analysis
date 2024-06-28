# DATA VERSION CONTROL STRATEGY:

We will maintain a single version of colorizer data for each segmented version of the dataset to avoid confusion about which version of the dataset is being loaded. To achieve this, we will have a single data directory for any segmented version of the dataset, which gets overwritten if we want to add or change the colorizer features or beahavior.

# RUNNING THE SCRIPT 

## RUNNING ON A NEW SET OF SEGMENTATIONS:
To create a new version of our colorized baseline data (which should only be done for segmentation updates) create a new data directory at `/allen/aics/animated-cell/Dan/fileserver/colorizer/data/{new_dataset_version_name}`.
Then create the data for each baseline colony for the colorizer by running:
```
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data/{new_dataset_version_name} --dataset small
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data/{new_dataset_version_name} --dataset medium 
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data/{new_dataset_version_name} --dataset large
```
These can be run in parallel.


To prepare the colony datasets to be grouped in a dropdown menu, create the file
`/allen/aics/animated-cell/Dan/fileserver/colorizer/data/{new_dataset_version_name}/collection.json`
containing the following:
```
[
    { "name": "Medium", "path": "medium" },
    { "name": "Large", "path": "large" },
    { "name": "Small", "path": "small" }
  ]
```

## OVERWRITING THE COLORIZER DATA FOR AN EXISTING DATASET (UNCHANGED SEGMENTATIONS/ROWS IN MANIFEST)
To overwrite an existing segmented version of the dataset (for example to add/remove/change a feature), simply run the script with the existing output directory and add the `--noframes` option to skip the frame generation step.

```
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data/{existing_dataset_version_name} --dataset small --noframes
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data/{existing_dataset_version_name} --dataset medium --noframes
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir /allen/aics/animated-cell/Dan/fileserver/colorizer/data/{existing_dataset_version_name} --dataset large --noframes
```

# PAST SEGMENTED DATASET VERSIONS
To load a dataset in the colorizer, go to `https://allen-cell-animated.github.io/nucmorph-colorizer/main/` and select the `Load` button in the upper right. Then enter the path of the JSON file for the baseline colony segmentation version you want to load. The existing options are:
V4: `https://dev-aics-dtp-001.int.allencell.org/dan-data/colorizer/data/nucmorph-v4/collection.json`
V3: `https://dev-aics-dtp-001.int.allencell.org/dan-data/colorizer/data/nucmorph-apr2024/collection.json`
V2: `https://dev-aics-dtp-001.int.allencell.org/dan-data/colorizer/data/nucmorph-reprocessed/collection.json`
V1: `https://dev-aics-dtp-001.int.allencell.org/dan-data/colorizer/data/nucmorph-updated/collection.json`