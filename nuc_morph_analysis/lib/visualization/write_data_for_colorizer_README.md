## Running on a new set of segmentations:
To create a new colorized  version of all subsets of the baseline data create a new data directory.
Then create the data for each dataset (subset of the data for all baseline colony data) by running:
```
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir {new_output_dir_name}
```
This will automatically create the datasets for all colonies in each of the datasets `baseline_colonies_dataset`, `full-interphase_dataset`, `lineage-annotated_dataset` and `exploratory_dataset`.
These can be run in parallel.


To prepare the colony datasets to be grouped in a dropdown menu, create the file
`{new_dataset_version_name}/collection.json` listing all colonies included in the dataset. For example, most of our subsets of data for the baseline colonies contain the following:
```
{
  "datasets": [
    { "name": "Small", "path": "small" },
    { "name": "Medium", "path": "medium" },
    { "name": "Large", "path": "large" }
  ],
  "metadata": {
    "name": "Baseline colonies dataset"
  }
}

```
while the "lineage-annotated" subset of the data only exists for the Small and Medium colonies, so the `collection.json` file for that dataset excludes the line pertaining to the Large colony.

## Overwriting the data for an existing dataset (unchanged segmentations/rows in manifest)
To overwrite an existing segmented version of the dataset (for example to add/remove/change a feature), simply run the script with the existing output directory and add the `--noframes` option to skip the frame generation step.

```
pdm run nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py --output_dir {existing_output_dir_name} --noframes
```