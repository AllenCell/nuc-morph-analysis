import numpy as np


def line(x, a, b):
    """
    This function gives the y values of a straight line
    of the form y = a * x + b

    Parameters
    ----------
    x: Number or ndarray
        The x values
    a: Number
        Slope of the line
    b: Number
        y-intercept of the line

    Returns
    -------
    Number or ndarray
        a * x + b
    """
    return a * x + b


def powerfunc(x, a, b, c):
    """
    This function gives the y values of power law
    of the form y = a * x + b

    Parameters
    ----------
    x: Number or ndarray
        The x values
    a: Number
        Slope of the line
    b: Number
        y-intercept of the line
    c: Number
        scaling of the line with x (c=1 gives linearity)

    Returns
    -------
    Number or ndarray
        a * x ** c + b
    """
    return b + a * (x**c)


def hyperbola(x, a1, a2, b1, b2, alpha):
    """
    Compute the value(s) at x of a hyperbola defined by a1, a2, b1, b2, and alpha.
    Specifically, computes the y values of the lower half of the hyperbola bounded
    above by two asymptotes.

    Parameters
    ----------
    x: Number or ndarray
        The x value(s) to evaluate at
    a1: Number
        Slope of asymptote 1
    b1: Number
        y-intercept of asymptote 1
    a2: Number
        Slope of asymptote 2
    b2: Number
        y-intercept of asymptote 2
    alpha: Number
        If the asymptotes intersect at (x0, y0), then the hyperbola goes through the point
        (x0, y0 - alpha).
        Alpha indirectly determines the foci and eccentricity of the hyperbola.

    Returns
    -------
    Number or ndarray
        Same shape as x, the y values for the hyperbola
    """
    y1 = line(x, a1, b1)
    y2 = line(x, a2, b2)
    return 0.5 * (y1 + y2 - np.sqrt((y1 - y2) ** 2 + alpha**2))


def hyperbola_jacobian(x, a1, a2, b1, b2, alpha):
    """
    Compute the Jacobian at x of the function `hyperbola`. Used for speeding up fitting the
    hyperbola to our data.

    Parameters
    ----------
    Same as hyperbola

    Returns
    -------
    ndarray
        The Jacobian of the hyperbola function at x. The value at [i, j] is the partial derivative
        of `hyperbola` with respect to i-th parameter (a1 is first), evaluated at x[j].
    """
    y1 = line(x, a1, b1)
    y2 = line(x, a2, b2)
    # r doesn't mean anything particular, it's just a common value across the partial derivatives
    r = (y1 - y2) / np.sqrt((y1 - y2) ** 2 + alpha**2)
    partial_a1 = 0.5 * x * (1 - r)  # partial derivative of `hyperbola` with respect to a1
    partial_b1 = 0.5 * (1 - r)
    partial_a2 = 0.5 * x * (1 + r)
    partial_b2 = 0.5 * (1 + r)
    partial_alpha = -0.5 * alpha / np.sqrt((y1 - y2) ** 2 + alpha**2)
    # Returned derivatives must be in the exact same order as the arguments of hyperbola
    return np.transpose([partial_a1, partial_a2, partial_b1, partial_b2, partial_alpha])
