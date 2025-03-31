# CCF: Cross-Correlation Functions Library

`CCF` is a Python library designed for deriving radial velocities (RV) and RV errors using cross-correlation functions (CCF). It provides tools for working with wavelength bins, calculating cross-correlation functions, and estimating radial velocities and their uncertainties.

## Features

- **Wavelength Binning**:
  - Linear and logarithmic wavelength grids.
  - Step size calculations for both linear and logarithmic grids.

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

## References
Tonry, J. & Davis, M. 1979, AJ, 84, 1511
