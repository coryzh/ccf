import numpy as np
import math
import constants as const


class WavelengthBin:
    def __init__(self, wave_min: float, wave_max: float, nbins: int):
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.nbins = nbins

        if self.wave_min >= self.wave_max:
            raise ValueError("wave_min must be smaller than wave_max.")

    @property
    def linear_grid(self) -> np.ndarray:
        return np.linspace(self.wave_min, self.wave_max, self.nbins + 1)

    @property
    def log_grid(self) -> np.ndarray:
        log_min = np.log(self.wave_min)
        log_max = np.log(self.wave_max)
        return np.logspace(log_min, log_max, self.nbins + 1, base=math.e)

    @property
    def linear_step(self) -> float:
        return self.linear_grid[1] - self.linear_grid[0]

    @property
    def log_step(self) -> float:
        """
        Step size in log space.
        Returns
        -------
        float
            The step size in log space.
        """
        return np.diff(np.log(self.log_grid))[0]


class NormalizedCCF:
    def __init__(self, data_obs: np.ndarray, data_temp: np.ndarray,
                 bins: WavelengthBin):
        self.data_obs = data_obs
        self.data_temp = data_temp
        self.bins = bins

    @property
    def lags(self) -> np.ndarray:
        return np.arange(-self.bins.nbins + 1, self.bins.nbins)

    @property
    def lags_in_kms(self) -> np.ndarray:
        self.lags * self.bins.log_step * const.c_in_kms


def test() -> None:
    bin_obj = WavelengthBin(6500, 6700, 100)
    print(len(bin_obj.linear_grid))


if __name__ == "__main__":
    test()
