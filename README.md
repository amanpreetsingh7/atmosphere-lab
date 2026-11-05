# Atmosphere Lab

Atmosphere Lab is a local Streamlit app for comparing stellar and planetary atmosphere spectra. It calculates the companion-to-star flux ratio as a function of wavelength, integrates it through a chosen response, and explores how model parameters, radius, and approximate stellar heating affect the contrast.

Developed by Amanpreet Singh. This is the standalone source distribution, based on the v0.8.10 workbench.

## Start the app

On macOS, open this folder as a project in PyCharm. Select a Python interpreter, run `install_requirements.py` in that interpreter, and then run `launch_app.py`. The browser opens at `http://127.0.0.1:8501`. Python 3.12 was used for this package's software checks; the requirements also retain compatibility constraints for the earlier macOS Python 3.9 setup.

From a terminal in the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python launch_app.py
```

The app starts without downloaded spectra. For a no-download calculation, run `python run_demo.py`. This writes a blackbody software demonstration under `results/`; it is not a native-atmosphere result. To compare atmosphere spectra, use **Model library** to install or import models locally. See [model downloads and source references](MODEL_DOWNLOADS.md) before starting a large download.

## Workflows

- **Spectra and contrast:** Select a star and a companion from a local catalog or upload compatible spectra. Enter their radii and run a comparison. Save comparisons and download plots, tables and provenance.
- **Filters and limits:** Use an explicit filter curve, or a clearly labeled idealized response for a method check. The app computes a ratio of separately integrated signals with the emitting-area factor. A supplied sensitivity curve is external input, not generated from the model.
- **Property predictions:** Open **Predict from stellar and companion properties** to select native grid parameters and enter separate filter and sky CSVs. Both curves and both spectra must cover the full band and use the same wavelength frame. The initial response is synthetic unity transmission, for demonstration only. Edit the paths and units in [the response template](templates/response_demo.json) to use a scientific band.
- **Parameter exploration:** Compare installed model values at fixed temperature, gravity and other held parameters. Cloud, composition and mixing controls depend on the model family and the installed grid cells. See [the parameter guide](PARAMETER_GUIDE.txt).
- **Separation and heating:** Explore fixed-temperature contrasts or use the approximate equilibrium heating option. In the property panel, the native contrast series can export temperature and radius sweeps, including optional heated separation curves. See [the contrast plot guide](CONTRAST_PLOT_GUIDE.md) for the main workbench workflow.
- **Physical properties:** Explore the installed evolution tables and conditional brightness or likelihood calculations. Results depend on the chosen model, band, radius and other assumptions.

## What is included

This repository contains the app source, model readers, numerical routines, installers for separately hosted model grids, synthetic examples, and software tests. It does not bundle downloaded spectra or archives, observing data, a reduction or injection pipeline, instrument-specific response files, previous results, or a user's local model catalog. Model and filter data are separate inputs; consult their original sources and cite them in scientific work.

The heating calculation combines intrinsic and equilibrium temperatures in fourth powers and uses isolated atmosphere spectra. It does not solve an irradiated atmosphere or the time-dependent two-layer thermal lag derivation. Interpolation is bounded by supported model cells; changing the grid, response or assumptions can change the result. The software checks use constructed inputs and do not establish observational accuracy or completeness.

## Run the software checks

```sh
python -m unittest discover -s tests -v
```

In PyCharm you can also run `run_laptop_validation.py` to generate a local report. For GitHub setup, see [upload instructions](UPLOAD_TO_GITHUB.md).
