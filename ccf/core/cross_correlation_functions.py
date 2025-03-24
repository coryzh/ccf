import numpy as np
import math
import ccf.constants as const


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

    def __repr__(self):
        return (f"Wavelengthbin(nbins={self.nbins}, "
                f"wave_min={self.wave_min:.1f}, "
                f"wave_max={self.wave_max:.1f})")


class NormalizedCCF:
    def __init__(self, data_obs: np.ndarray, data_temp: np.ndarray,
                 bins: WavelengthBin):
        self.data_obs = data_obs
        self.data_temp = data_temp
        self.bins = bins

        if len(data_obs) != len(data_temp):
            raise ValueError(
                f"data_obs, length: {len(data_obs)} and \
                data_temp, length: {len(data_temp)} must have the same length."
            )

    @property
    def series_length(self) -> int:
        return len(self.data_obs)

    @property
    def lags(self) -> np.ndarray:
        return np.arange(-self.bins.nbins + 1, self.bins.nbins)

    @property
    def lags_in_kms(self) -> np.ndarray:
        return self.lags * self.bins.log_step * const.c_in_kms

    @property
    def rms_obs(self) -> float:
        return np.sqrt(np.sum(self.data_obs ** 2) / self.bins.nbins)

    @property
    def rms_temp(self) -> float:
        return np.sqrt(np.sum(self.data_temp ** 2) / self.bins.nbins)

    def normalized_ccf(self) -> np.ndarray:
        corr = np.zeros(len(self.lags))
        for i, lag in enumerate(self.lags):
            if lag >= 0:
                t_min, t_max = 0, self.series_length - lag
            else:
                t_min, t_max = -lag, self.series_length

            corr_list = [
                self.data_obs[t] * self.data_temp[t + lag]
                for t in range(t_min, t_max)
            ]
            corr[i] = np.sum(np.array(corr_list))

        corr /= self.bins.nbins * self.rms_obs * self.rms_temp
        return corr

    @property
    def primary_peak(self) -> int:
        peak_id = np.argmax(self.normalized_ccf())
        return self.lags[peak_id]

    @property
    def primary_peak_in_kms(self) -> float:
        peak_id = np.argmax(self.normalized_ccf())
        return self.lags_in_kms[peak_id]


def test() -> None:
    bin_obj = WavelengthBin(6500, 6700, 100)
    print(len(bin_obj.linear_grid))


if __name__ == "__main__":
    test()
