# Iono-LuGRE

[![Interactive Dashboard](https://img.shields.io/badge/3D-GNSS%20raypath%20&%20plasma%20simulations-191970)](https://viventriglia.github.io/iono-lugre/)

Official repository for the paper <i>Observing the Earth’s Plasmasphere and Ionosphere from the Lunar Surface</i> (Cesaroni et al. 2026, submitted to <i>Geophysical Research Letters</i>).
Read the preprint here: </b> https://doi.org/10.22541/essoar.176902858.83882772/v1</b>

## Overview

Surrounding our planet is a vast layer of charged particles called the <b>ionosphere/plasmasphere system</b>, which plays a critical role in <b>Space Weather</b>. Despite its importance, a significant <b>blind spot</b> exists between the top of our atmosphere and deep-space altitudes where Low Earth Orbit (LEO) satellites cannot observe the plasma layers above them.

This project leverages data from the <b>Lunar GNSS Receiver Experiment</b> (<b>[LuGRE](https://www.nasa.gov/centers-and-facilities/goddard/nasa-and-italian-space-agency-test-future-lunar-navigation-technology/)</b>), a pioneering mission that used the Moon as a platform to perform limb sounding of the Earth’s environment. By tracking GPS and Galileo signals from the lunar surface, LuGRE bridges observational gaps at altitudes exceeding 3,000 km.

<p align="center"> <img src="assets/iono_lugre.webp" width="800" alt="LuGRE System Visualization">
<em>Earth, Plasmasphere, a GNSS satellite, and the Moon (not drawn to scale).</em> </p>

## Key Findings

- <b>Refilling Efficiency</b>: LuGRE data indicates a lower plasma refilling efficiency than currently parameterized in standard models, e.g. Global Core Plasma Model (GCPM).

- <b>Lunar Observatory</b>: This project demonstrates that the Moon is an ideal location for permanent Space Weather observatories improving safety of space technology.

## Corrections

<b>v0.2.0 (November 2026).</b> The GCPM slant-TEC curves in Figure 3 of the preprint were affected by two bugs in PyGCPM:

- IRI's solar-index arrays overflowed with a current `ig_rz.dat`, so IG12 was read as about 10 instead of about 113.
- The Python wrapper passed the year as the day of year, so every epoch was modelled as 25 January 2027.

The LuGRE observations are not affected. This release pins a fixed PyGCPM fork ([mguerra96/PyGCPM@f8464a4](https://github.com/mguerra96/PyGCPM/commit/f8464a4a58983356144b36208a552b5f5cd74a01)), which also ships the current IRI index files (`ig_rz.dat` dated 19 Dec 2025, and `ap.dat`). The corrected curves and the full analysis are in [`fig3_recomputed/`](fig3_recomputed/REPORT.md). We thank the external researcher who reported the discrepancy.

The interactive 3D plots in `docs/` were produced with the earlier PyGCPM build and have not yet been regenerated.

## Repository Structure

```bash
.
├── assets/               # Images, etc.
├── docs/                 # GitHub Pages source (interactive HTML plots)
├── fig3_recomputed/      # Corrected Figure 3 GCPM curves and the bug report (v0.2.0)
├── src/                  # Functions for the processing and plotting of LuGRE data
├── GCPM_vs_LuGRE.ipynb   # Notebook to reproduce paper results
├── LICENSE               # GNU GPL license
├── requirements.txt      # Dependencies
└── README.md
```

## Data Access

The pre-processed GNSS dataset is available on [Zenodo](https://zenodo.org/records/18220778).
