import numpy as np
from ccf.core.cross_correlation_functions import WavelengthBin


def interp_resampler(x: np.ndarray, y: np.ndarray,
                     bins: WavelengthBin, **kwargs) -> np.ndarray:
    y_interp = np.interp(bins.log_grid, x, y, **kwargs)

    return y_interp
