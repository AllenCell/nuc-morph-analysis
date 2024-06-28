import imageio.v3 as iio
import os
import shutil
import stat

from pytest import fixture
from nuc_morph_analysis.lib.visualization import movie_tools

TEST_DIR = os.path.abspath("test_data")


@fixture(autouse=True, scope="session")
def generate_frames():
    # setup at beginning of this module
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    # Arrange
    os.makedirs(TEST_DIR, exist_ok=True)
    for i, frame in enumerate(iio.imiter("imageio:cockatoo.mp4", plugin="pyav")):
        filename = os.path.join(TEST_DIR, f"img{i:03}.png")
        iio.imwrite(filename, frame)


def is_regular_file(path):
    statinfo = os.stat(path)
    return stat.S_ISREG(statinfo.st_mode) != 0


def test_make_gif():
    # Act
    movie_tools.make_gif(TEST_DIR, TEST_DIR, "test_make_gif", fps=20)

    # Assert
    expected_file = os.path.join(TEST_DIR, "test_make_gif_20fps.gif")
    assert is_regular_file(expected_file)
    # Optionally, you could watch the output to confirm it looks as expected


def test_make_mp4():
    # Act
    movie_tools.make_mp4(TEST_DIR, TEST_DIR, "test_make_mp4", fps=20)

    # Assert
    expected_file = os.path.join(TEST_DIR, "test_make_mp4_20fps.mp4")
    assert is_regular_file(expected_file)
    # Optionally, you could watch the output to confirm it looks as expected
