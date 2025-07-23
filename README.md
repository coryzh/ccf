# CCF: Cross-Correlation Functions Library

`CCF` is a Python library designed for deriving radial velocities (RV) and RV errors using cross-correlation functions (CCF). It provides tools for working with wavelength bins, calculating cross-correlation functions, and estimating radial velocities and their uncertainties.

## Features

- **Wavelength Binning**:
  - Generate both linear and logarithmic wavelength grid given minimum and maximum wavelength

- **Resampling**:
  - Resampling using linear interpolation.

- **Cross-Correlation Functions**:
  - Compute cross-correlation between observed and template spectra.
  - Detect peaks in the cross-correlation function.
  - Calculate radial velocities and their uncertainties.

- **Error Estimation**:
  - Compute the r-ratio using the [Tonry & Davis (1979)](https://ui.adsabs.harvard.edu/abs/1979AJ.....84.1511T/abstract) method.
  - Estimate RV uncertainties based on the closest neighbouring peak.

## Installation

To install the library, clone the repository and install the dependencies:

```bash
pip install git+https://github.com/coryzh/ccf.git
```

## Usage

As an example,

```python
import numpy as np
from ccf.core.cross_correlation_functions import NormalizedCCF, WavelengthBin
from ccf.core.resampler import interp_resampler
# Step 1: Instantiate a WavelengthBin object; this will be the wavelength range
# of interest.
wave_min = 4000  # Minimum wavelength in Angstroms
wave_max = 7000  # Maximum wavelength in Angstroms
nbins = 100  # Number of bins
wavelength_bin = WavelengthBin(wave_min, wave_max, nbins)

# Step 2: Load the observed and template spectral data. E.g., they can be 
# loaded from ASCII or .fits files. Suppose that the .txt files have one column
# for wavelength and another for fluxes.
wave_obs, flux_obs = np.loadtxt("observed_spectrum.txt", unpack=True)
wave_temp, flux_temp = np.loadtxt("template_spectrum.txt", unpack=True)

# Step 3: Make sure both spectra are sampled onto the same wavelength bins.
flux_obs_resampled = interp_resampler(wave_obs, flux_obs, wavelength_bin)
flux_temp_resampled = interp_resampler(wave_temp, flux_temp, wavelength_bin)

# Step 4: Instantiate a NormalizedCCF object.
ccf = NormalizedCCF(
  data_obs=flux_obs_resampled,
  data_temp=flux_temp_resampled,
  bins=wavelength_bin
)

# Now RV and RV uncertainty can be easily accessed by .rv and .rv_err.
print(f"RV: {ccf.rv:.2f} +/- {ccf.rv_err:.2f} km/s")
```


## References
Tonry, J. & Davis, M. 1979, AJ, 84, 1511
