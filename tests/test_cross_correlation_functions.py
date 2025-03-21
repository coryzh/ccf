import pytest
import numpy as np
from ccf.core.cross_correlation_functions import WavelengthBin


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
