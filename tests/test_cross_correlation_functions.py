import pytest
import numpy as np
from ccf.core.cross_correlation_functions import WavelengthBin, NormalizedCCF
from astropy.modeling.models import Lorentz1D


class TestWavelengthBin:
    def test_initialization_valid(self):
        bin_obj = WavelengthBin(6500., 6700., 100)
        assert bin_obj.wave_min == 6500
        assert bin_obj.wave_max == 6700
        assert bin_obj.nbins == 100

    def test_initialization_invalid(self):
        with pytest.raises(ValueError) as excinfo:
            WavelengthBin(20.0, 10.0, 5)
        assert str(excinfo.value) == "wave_min must be smaller than wave_max."

    def test_linear_grid(self):
        bin_obj = WavelengthBin(6500., 6700., 100)
        linear_grid = bin_obj.linear_grid
        assert len(linear_grid) == 100 + 1
        assert np.isclose(linear_grid[0], 6500)
        assert np.isclose(linear_grid[-1], 6700)
        assert np.allclose(np.diff(linear_grid), bin_obj.linear_step)

    def test_log_grid(self):
        bin_obj = WavelengthBin(6500, 6700, 100)
        log_grid = bin_obj.log_grid
        assert len(log_grid) == 100 + 1
        assert np.isclose(log_grid[0], 6500.)
        assert np.isclose(log_grid[-1], 6700.)

    def test_linear_step(self):
        bin_obj = WavelengthBin(6500., 6700., 100)
        linear_step = bin_obj.linear_step
        assert np.isclose(linear_step, (6700. - 6500.) / 100)

    def test_log_step(self):
        bin_obj = WavelengthBin(6500., 6700., 100)
        log_step = bin_obj.log_step
        log_grid = bin_obj.log_grid
        expected_log_step = np.log(log_grid[5]) - np.log(log_grid[4])
        assert np.isclose(log_step, expected_log_step)


class TestNormalizedCCF:
    def setup_method(self):
        self.nbins = 5
        self.wave_centre = 6564.6
        self.window = 5
        self.shift = 1
        self.bins = WavelengthBin(
            self.wave_centre - 0.5 * self.window,
            self.wave_centre + 0.5 * self.window,
            self.nbins
        )

        line_profile = Lorentz1D(
            x_0=self.wave_centre,
            fwhm=5.0, amplitude=1.0
        )

        line_profile_shifted = Lorentz1D(
            x_0=self.wave_centre-self.shift, fwhm=5.0, amplitude=1.0
        )

        noise = np.random.randn(self.nbins + 1) * 0.1

        self.data_obs = line_profile_shifted(self.bins.linear_grid) + noise
        self.data_temp = line_profile(self.bins.linear_grid)
        self.ccf = NormalizedCCF(self.data_obs, self.data_temp, self.bins)

    def test_initialization_invalid(self):
        with pytest.raises(ValueError):
            NormalizedCCF(
                np.array([1, 2, 3]), np.array([1, 2, 3, 4]),
                self.bins
            )

    def test_series_length(self):
        assert self.ccf.series_length == self.nbins + 1

    def test_lags(self):
        expected_lags = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        assert np.array_equal(self.ccf.lags, expected_lags)

    def test_lags_in_kms(self):
        # const = type("const", (object,), {"c_in_kms": 299792.458})
        expected_lags_kms = self.ccf.lags * self.bins.log_step * 299792.458
        assert np.allclose(self.ccf.lags_in_kms, expected_lags_kms)

    def test_rms_obs(self):
        expected_rms_obs = np.sqrt(
            np.sum(self.data_obs ** 2) / self.bins.nbins
        )

        assert np.isclose(self.ccf.rms_obs, expected_rms_obs)

    def test_rms_temp(self):
        expected_rms_temp = np.sqrt(
            np.sum(self.data_temp ** 2) / self.bins.nbins
        )

        assert np.isclose(self.ccf.rms_temp, expected_rms_temp)

    def test_length_normalized_ccf(self):
        ccf_result = self.ccf.normalized_ccf()
        assert len(ccf_result) == len(self.ccf.lags)
