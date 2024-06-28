import matplotlib.pyplot as plt
import os
import shutil
import stat

from nuc_morph_analysis.lib.visualization import notebook_tools


def is_regular_file(path):
    statinfo = os.stat(path)
    return stat.S_ISREG(statinfo.st_mode) != 0


TEST_DIR = os.path.abspath("test_data")


def test_save_and_show_plot():
    """
    Regression test.
    Expected file outputs are:
      0.png
      1.png
      2.png
    Before this bug was fixed, this test produced the following files:
      0.png
      11.png (copy of 0.png)
      12.png (should be 1.png)
      21.png (copy of 0.png)
      22.png (copy of 12.png)
      23.png (should be 2.png)
    """
    # Arrange
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    dummy_data = [1, 2, 3, 4]

    # Act
    for i in range(3):
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        plt.scatter(dummy_data, dummy_data)
        notebook_tools.save_and_show_plot(f"{TEST_DIR}/figures/{i}", file_extension=".png")

    # Assert
    for i in range(3):
        assert is_regular_file(f"{TEST_DIR}/figures/{i}.png")

    # Clean up
    shutil.rmtree(TEST_DIR, ignore_errors=True)
