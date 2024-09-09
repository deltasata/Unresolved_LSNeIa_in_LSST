# Detecting Unresolved Lensed SNe Ia in LSST Using Blended Light Curves

This repository contains code (`ipython notebooks`) designed to calculate the number of resolved and unresolved lensed Type Ia supernovae (SNe Ia) expected in the Legacy Survey of Space and Time (LSST) by the *Vera Rubin Observatory*, as well as the observed blended light curves for the unresolved systems. For a detailed explanation, please refer to [arXiv:2404.15389](https://arxiv.org/abs/2404.15389).

## Data Requirements

To run the provided notebooks, download the necessary datasets from [Zenodo](https://zenodo.org/records/13644602). Detailed instructions are given below.

### 1. `unresolved_stat`

This notebook analyzes the statistics of both resolved and unresolved lensed Type Ia supernovae expected in 10 effective year of LSST run. It also explores the properties of unresolved systems (e.g., redshift, time delay, magnification) and compares them with resolved systems.

- **Required Data:** Download `unresolved_catalog.zip` from [Zenodo](https://zenodo.org/records/13644602) and place it's contents in the same directory as the notebook.

### 2. `build_blended_lc`

This directory contains notebooks to simulate multi-band LSST-like blended light curves for unresolved lensed SNe Ia considering `baseline v3.2 observing strategy`. It includes the following:

- **A) `sata_build_unresolved.ipynb`:** Generates light curves for unresolved doubles, quads, and unlensed SNe Ia based on `baseline v3.2 observing strategy`. These light curves correspond to the controlled sets A, B, and C from [arXiv:2404.15389](https://arxiv.org/abs/2404.15389). You can adjust the number of doubles, quads, and unlensed systems for training purposes.
  
- **B) `LSST_set_sata_build_unresolved_just_over_pop.ipynb`:** Creates mock light curves for unresolved LSST samples, referred to as "set LSST" in [arXiv:2404.15389](https://arxiv.org/abs/2404.15389).

- **Required Data:** Download `build_blended_lc.zip` from [Zenodo](https://zenodo.org/records/13644602) and place it's contents in the same directory as the notebooks.
