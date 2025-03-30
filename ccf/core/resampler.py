import numpy as np
from ccf.core.cross_correlation_functions import WavelengthBin


def interp_resampler(x: np.ndarray, y: np.ndarray,
                     bins: WavelengthBin, **kwargs) -> np.ndarray:
    """
    Return the interpolated values given arrays of x and y.

    Parameters
    ----------
    x : np.ndarray
        Values of the independent variable.
    y : np.ndarray
        Values of the depedent variable.
    bins : WavelengthBin
        Wavelength bins over which values are interpolated.
    kwargs : dict
        Keyword argument of the numpy.interp() method.
    Returns
    -------
    np.ndarray
        Interpolated values.
    """
    y_interp = np.interp(bins.log_grid, x, y, **kwargs)

    return y_interp
