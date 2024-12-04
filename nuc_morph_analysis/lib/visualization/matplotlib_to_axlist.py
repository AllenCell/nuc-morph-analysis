from typing import List, Union, cast
from matplotlib.axes import Axes
from numpy.typing import NDArray

def type_axlist(plt_axis_np: Union[Axes, NDArray]) -> List[Axes]:
    """
    Figure.subplots returns Axes or array of Axes, but it's improperly typed
    See discussion here:
    https://github.com/numpy/numpy/issues/24738

    Example:
    >>> fig,axlist_untyped = plt.subplots(nrows=2,ncols=1,figsize=(6.5,8))
    >>> axlist: List[Axes] = type_axlist(axlist_untyped)
    """
    # This function was adapted from this PR, licensed under BSD-3:
    # https://github.com/pytorch/captum/pull/1416/files
    plt_axis_list: List[Axes] = []
    if type(plt_axis_np) == Axes:
        plt_axis_list = [plt_axis_np]
    else:
        plt_axis_list = cast(List[Axes], cast(NDArray, plt_axis_np).tolist())
    return plt_axis_list