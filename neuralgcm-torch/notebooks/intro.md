# neuralgcm-torch

[NeuralGCM](https://github.com/neuralgcm/neuralgcm) is a hybrid ML + physics
global circulation model that pairs a differentiable spectral dynamical core
with learned physics to forecast weather and run climate-scale simulations,
originally written in JAX. **`neuralgcm-torch` brings it to PyTorch** — load the
published NeuralGCM checkpoints (converted to `torch`) and forecast in a few
lines, with no JAX, gin or haiku at runtime.

```{video} media/forecast_0_7_deg.mp4
:width: 520
:autoplay:
:loop:
:muted:
:playsinline:
:align: center
```

<p align="center"><em>A 12-day NeuralGCM-0.7° forecast — 850&nbsp;hPa specific
humidity — on a slowly rotating globe.</em></p>

These pages are the project's example notebooks, rendered from their committed
outputs. For the full API and design notes, see the
[README on GitHub](https://github.com/DSIP-FBK/neuralgcm-torch/tree/main/neuralgcm-torch).

```{admonition} Not affiliated with NeuralGCM or Google
:class: note
This is an independent PyTorch reimplementation built on the original team's
published research and open-source weights. All credit for the models and
science goes to them — see the
[NeuralGCM repository](https://github.com/neuralgcm/neuralgcm) and please cite
*Kochkov et al., "Neural general circulation models for weather and climate",
Nature 632 (2024).*
```

## Running these notebooks yourself

Download any notebook with the button at the top-right of its page, then run it
against an environment with the package installed — see
[Installation](installation) for the PyPI and clone-and-run setups. A CUDA GPU
and network access are recommended; the higher-resolution and climate-stability
runs assume a GPU.

## The notebooks

- **Forecasting** — `forecast_quickstart` (the 2.8° ERA5 walkthrough) and the
  1.4° / 0.7° / ensemble / precipitation / evaporation variants.
- **Climate** — `climate_stability` drives the stable models far past weather
  lead times (months to years) with seasonal ERA5 forcing.
- **Data & model internals** — regridding, model internals and autograd, and
  editing the converted config.

The original NeuralGCM team's pre-computed simulation outputs (the
`gs://neuralgcm/…` Zarr stores) are catalogued in the upstream
[NeuralGCM simulation datasets](https://neuralgcm.readthedocs.io/en/latest/neuralgcm_datasets.html)
documentation.

Use the sidebar to open any notebook.
