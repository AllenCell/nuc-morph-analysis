# Colony context and size-dependent compensation mechanisms give rise to variations in nuclear growth trajectories
The code in this repository generates all of the figures for Dixon et al. 2024 [(bioRxiv)](https://www.biorxiv.org/content/10.1101/2024.06.28.601071v1). It is primarily intended to support reproducibility of our research. In addition, researchers may find parts of this code valuable for future work.


For a description of the cell treatments, imaging, and the purpose of each analysis, please refer to the paper.

The data used in this analysis are publicly available on [Quilt](https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/) under the [Allen Insitute for Cell Science Terms of Use](https://www.allencell.org/terms-of-use.html). The data are also available via the AWS S3 API directly in the folder `s3://allencell/aics/nuc-morph-dataset`.

## Installation
> [!NOTE]
> These are the basic installation steps. However, our recommendation is to install with `pyenv` and `pdm`. See advanced installation instructions [here](docs/INSTALL.md).

1. Install Python 3.9 and `git`.
2. Clone this git repository.
```bash
git clone git@github.com:AllenCell/nuc-morph-analysis.git
cd nuc-morph-analysis
```
3. Create a new virtual environment and activate it.
```bash
python -m venv venv
source venv/bin/activate
```
4. Install the required packages for your operating system. Replace `linux` with `macos` or `windows` as appropriate.
```bash
pip install -r requirements/linux/requirements.txt
```

## Reproduce figures

List available workflows with the following command.
```bash
python run_all_manuscript_workflows.py --list
```

Use the `--only` flag to run any one workflow. Confirm that your installation is working by running a fast workflow.
```bash
python run_all_manuscript_workflows.py --only error
```
This should write figures to the `nuc-morph-analysis/nuc_morph_analysis/analyses/error_morflowgenesis/figures/` directory.

> [!IMPORTANT]
> Most workflows provided here are designed to run in a high-performance computing setting. They use 30-60GB of RAM and running all of them will take many hours even with a fast machine.

To run all the analyses in the paper, omit any options.
```bash
python run_all_manuscript_workflows.py
```
Figures are saved to the directories `nuc-morph-analysis/nuc_morph_analysis/analyses/*/figures/`.


## Reproduce track pre-processing

In addition to `run_all_manuscript_workflows.py`, this repository includes a few other entrypoints for specific pre-processing tasks.

* To reproduce the [hiPSC single nuclei timelapse analysis datasets](https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/hipsc_single_nuclei_timelapse_analysis_datasets/), run:
```bash
python nuc_morph_analysis/lib/preprocessing/save_datasets_for_quilt.py
```
* To reproduce [2024-06-25_baseline_intermediate_manifest.parquet](https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/supplemental_files/intermediate_manifests/2024-06-25_baseline_intermediate_manifest.parquet), run:
```bash
python nuc_morph_analysis/lib/preprocessing/generate_main_manifest.py
```
* To reproduce the [feeding control and inhibitor intermediate manifests](https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/supplemental_files/intermediate_manifests/), run:
```bash
python nuc_morph_analysis/lib/preprocessing/generate_perturbation_manifest.py
```
* To reproduce the [timelapse feature explorer datasets](https://open.quiltdata.com/b/allencell/tree/aics/nuc-morph-dataset/timelapse_feature_explorer_datasets/), run:
```bash
python nuc_morph_analysis/lib/visualization/write_data_for_colorizer.py
```
