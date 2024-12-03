from typing import List, Union, cast
from matplotlib.axes import Axes
from numpy.typing import NDArray

def type_axlist(plt_axis_np: Union[Axes, NDArray]) -> List[Axes]:
        # Figure.subplots returns Axes or array of Axes
        # https://github.com/numpy/numpy/issues/24738
    plt_axis_list: List[Axes] = []
    if type(plt_axis_np) == Axes:
        plt_axis_list = [plt_axis_np]
    else:
        plt_axis_list = cast(List[Axes], cast(NDArray, plt_axis_np).tolist())
    return plt_axis_list