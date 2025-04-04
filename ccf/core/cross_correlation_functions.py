import numpy as np
import math
import ccf.constants as const
from scipy.signal import find_peaks
from typing import Tuple


class WavelengthBin:
    def __init__(self, wave_min: float, wave_max: float, nbins: int):
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.nbins = nbins
        self.ndata = nbins + 1

        if self.wave_min >= self.wave_max:
            raise ValueError("wave_min must be smaller than wave_max.")

    @property
    def linear_grid(self) -> np.ndarray:
        return np.linspace(self.wave_min, self.wave_max, self.ndata)

    @property
    def log_grid(self) -> np.ndarray:
        log_min = np.log(self.wave_min)
        log_max = np.log(self.wave_max)
        return np.logspace(log_min, log_max, self.ndata, base=math.e)

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
                f"ndata={self.ndata}, "
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
        """
        Lags between the template and the observed spectra. It is in units of
        logarithmic wavelength bin width. A positive lag means that the
        observed spectrum is red-shifted from the template spectrum. This
        definition makes sure that lags have the same sign as radial
        velocities.

        Returns
        -------
        np.ndarray
            Lags as number of wavelength bins.
        """
        return np.arange(-self.bins.ndata + 1, self.bins.ndata)

    @property
    def lags_in_kms(self) -> np.ndarray:
        """
        Lags in units of km/s. It is computed by
            lags [km/s] = lags [log] * log_step_size * c [km/s]
        Returns
        -------
        np.ndarray
            Lags in units of km/s.
        """
        return self.lags * self.bins.log_step * const.c_in_kms

    @property
    def rms_obs(self) -> float:
        """
        Root mean square (RMS) of the observed spectrum.

        Returns
        -------
        float
            RMS value of the observed spectrum
        """
        return np.sqrt(np.sum(self.data_obs ** 2) / self.bins.ndata)

    @property
    def rms_temp(self) -> float:
        """
        Root mean square (RMS) of the template spectrum.
        Returns
        -------
        float
            RMS value of the template spectrum.
        """
        return np.sqrt(np.sum(self.data_temp ** 2) / self.bins.ndata)

    def ccf(self) -> np.ndarray:
        """
        N.B. in np.correlate, the first series is to be shifted while the
        second one is kept fixed. In our definition the observed spectrum
        is shifted against the template for a range of lag values.
        Returns
        -------
        np.ndarray
            Cross-correlation function values.
        """
        corr = np.correlate(self.data_obs, self.data_temp, mode="full")
        corr /= self.rms_obs * self.rms_temp * self.bins.ndata
        return corr

    def ccf_peaks(self, min_height: float = 0.001) -> Tuple:
        """
        Returns the peak locations and heights, both positive and negative
        of the cross-correlation function.

        Parameters
        ----------
        min_height : float, optional
            The minimum of absolute peak height used for peak detection, by
            default 0.001

        Returns
        -------
        Tuple
            Peak location (integer indices), and peak heights (float).
        """
        ccf = self.ccf()
        pos_peak, _ = find_peaks(ccf, height=None, prominence=0)
        neg_peak, _ = find_peaks(-ccf, height=None, prominence=0)
        pos_peak_heights = np.abs(ccf[pos_peak])
        neg_peak_heights = np.abs(ccf[neg_peak])

        pos_peak_valid = pos_peak[np.where(pos_peak_heights >= min_height)[0]]
        neg_peak_valid = neg_peak[np.where(neg_peak_heights >= min_height)[0]]

        peak_indices = np.concatenate((pos_peak_valid, neg_peak_valid))
        peak_indices = np.sort(peak_indices)

        return peak_indices, ccf[peak_indices]

    @property
    def primary_peak_loc(self) -> int:
        """
        Index location of the highest peak.
        Returns
        -------
        int
            Integer index of the highest peak.
        """
        peak_id = np.argmax(self.ccf())
        return self.lags[peak_id]

    @property
    def rv(self) -> float:
        """
        Radial velocity of the observed spectrum relative to the template.
        Returns
        -------
        float
            Radial velocity (in km/s) of the observed spectrum.
        """
        peak_indices, heights = self.ccf_peaks()
        max_peak_id = peak_indices[np.argmax(heights)]

        return self.lags_in_kms[max_peak_id]

    @property
    def primary_peak_height(self) -> float:
        ccf = self.ccf()
        return ccf[self.primary_peak_loc + self.bins.ndata - 1]

    def antisymmetric(self, lag_0: int) -> np.ndarray:
        i_min = abs(lag_0)
        i_max = 2 * self.bins.ndata - 2 - abs(lag_0)

        if i_min >= i_max:
            raise ValueError(
                f"Invalid value for lag_0. Given the bins, lag_0 can only "
                f"take a value between -{self.bins.ndata - 1} "
                f"and {self.bins.ndata - 1}."
            )

        antisymmetric = np.zeros_like(self.ccf())
        ccf = self.ccf()
        # Only element within i_min and i_max are computed, the others are kept
        # at 0.
        for i in range(i_min, i_max + 1):
            antisymmetric[i] = (
                ccf[i - self.bins.ndata + 1 + lag_0]
                - ccf[-i + self.bins.ndata - 1 + lag_0]
            )

        return antisymmetric

    def rms_antisymmetric(self, lag_0: int) -> float:
        a = self.antisymmetric(lag_0=lag_0)
        sigma_a = np.sqrt(np.sum(a ** 2) / (2 * self.bins.ndata))
        return sigma_a

    @property
    def r_ratio(self) -> float:
        """
        The r-ratio from Tonry & Davis 1979.
        Returns
        -------
        float
            The r-ratio given the primary peak height and the rms of the
            antisymmetric component of the CCF.
        """
        r = (
            self.primary_peak_height
            / (np.sqrt(2) *
               self.rms_antisymmetric(lag_0=self.primary_peak_loc))
        )

        return r

    @property
    def averaged_inter_peak_distance(self) -> float:
        peak_locs = np.sort(self.ccf_peaks()[0])
        peak_sep = np.diff(peak_locs)

        return np.mean(peak_sep)

    @property
    def rv_err(self) -> float:
        """
        Uncertainty on the radial velocity estimate. It is estimated using the
        Tonry & David 1979 method.

        Returns
        -------
        float
            Uncertainty on the radial velocity (in km/s).
        """
        result = (
            0.25 * self.averaged_inter_peak_distance * (1 / (1 + self.r_ratio))
            * self.bins.log_step * const.c_in_kms
        )

        return result
