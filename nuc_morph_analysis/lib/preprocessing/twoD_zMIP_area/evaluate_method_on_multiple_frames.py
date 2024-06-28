# %%
from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import evaluate_method
from multiprocessing import Pool, cpu_count
from itertools import product
from tqdm import tqdm
import numpy as np


def run_task_inparallel(dataset_time):
    run_task(dataset_time[0], dataset_time[1])


def run_task(dataset, time):
    evaluate_method.evaluate_method(dataset, time)


def run_script(
    dataset_list,
    time_list=[10],
):
    dataset_time_list = list(product(dataset_list, time_list))
    print("running evaluate method on multiple frames in parallel")
    with Pool(cpu_count()) as p:
        for _ in tqdm(
            p.imap_unordered(run_task_inparallel, dataset_time_list), total=len(dataset_time_list)
        ):
            pass


if __name__ == "__main__":
    run_script(dataset_list=["drug_perturbation_2_scene6"], time_list=np.arange(93, 99))

# %%
