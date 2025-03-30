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
  - Compute the r-ratio using the Tonry & Davis (1979) method.
  - Estimate RV uncertainties based on the closest neighbouring peak.

## Installation

To install the library, clone the repository and install the dependencies:

```bash
pip install git+https://github.com/your-username/your-repository-name.git
```

## Acknowledgments
This library is inspired by the methods described in Tonry & Davis (1979) for cross-correlation and RV error estimation.
