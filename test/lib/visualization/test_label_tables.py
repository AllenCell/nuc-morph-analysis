#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Docs: https://docs.pytest.org/en/latest/example/simple.html
    https://docs.pytest.org/en/latest/plugins.html#requiring-loading-plugins-in-a-test-module-or-conftest-file
"""
import pytest
import numpy as np
from nuc_morph_analysis.lib.visualization.plotting_tools import get_plot_labels_for_metric
from nuc_morph_analysis.lib.preprocessing.system_info import PIXEL_SIZE_YX_100x


@pytest.mark.parametrize(
    "input_metric, dataset, expected_output",
    [
        (
            "height",
            "all_baseline",
            (
                PIXEL_SIZE_YX_100x,
                "Height",
                "(μm)",
            ),
        ),
        (
            "volume",
            "all_baseline",
            (
                PIXEL_SIZE_YX_100x**3,
                "Volume",
                "(μm\u00B3)",
            ),
        ),
    ],
)
def test_label_tables(input_metric, dataset, expected_output):
    scale, label, unit, _ = get_plot_labels_for_metric(input_metric, dataset)
    assert np.isclose(scale, expected_output[0])
    assert label == expected_output[1]
    assert unit == expected_output[2]
