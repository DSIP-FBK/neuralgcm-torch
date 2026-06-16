# Notebooks

End-to-end examples for [**neuralgcm-torch**](../README.md) — the
idiomatic-PyTorch rewrite of NeuralGCM. Each notebook fetches the checkpoint
it needs from the Hugging Face Hub on first run (cached via
`pretrained.fetch_checkpoint`), or reuses a local `checkpoints/` copy if
present, so they run as-is. A CUDA GPU is recommended; the coarse models also
run on CPU. Install the Hub extra for downloads: `pip install
'neuralgcm-torch[hub]'`.

## Getting started

- **[forecast_quickstart.ipynb](forecast_quickstart.ipynb)** — Start here. A
  4-day ERA5 weather forecast with the 2.8° deterministic model, ending in a
  forecast-vs-ERA5 comparison. The PyTorch port of the upstream
  `inference_demo`.

## Forecasting the published checkpoints

- **[forecast_1_4_deg.ipynb](forecast_1_4_deg.ipynb)** — The same forecast with
  the **1.4° deterministic** model (TL127 core, 256×128 grid, 18.3M params) —
  the middle of the deterministic family.
- **[forecast_0_7_deg.ipynb](forecast_0_7_deg.ipynb)** — The flagship **0.7°
  deterministic** model (NeuralGCM-0.7: TL255 core, 512×256 grid, 31M params) —
  the most accurate at short lead times.
- **[forecast_ens_1_4_deg.ipynb](forecast_ens_1_4_deg.ipynb)** — Ensemble
  forecasting with the **stochastic 1.4°** model (NeuralGCM-ENS): batched
  members from different seeds, with ensemble spread and ensemble-mean skill.
- **[forecast_precip_2_8_deg.ipynb](forecast_precip_2_8_deg.ipynb)** — The
  **2.8° stochastic precipitation** model (trained against satellite
  precipitation), which adds cumulative-precipitation outputs.
- **[forecast_evap_2_8_deg.ipynb](forecast_evap_2_8_deg.ipynb)** — The **2.8°
  stochastic evaporation** variant, where precipitation follows from the water
  budget; better behaved at sub-6-hour timescales.

## Going deeper

- **[data_preparation.ipynb](data_preparation.ipynb)** — Preparing ERA5-style
  data: conservative regridding and converting between `xarray.Dataset` and the
  dict-of-tensors format the model API uses.
- **[deepdive_into_models.ipynb](deepdive_into_models.ipynb)** — The internals
  of a trained model: the encoder / learned-physics / dynamical-core structure,
  the encoded state, autograd through the model, and the stochastic fields.
  Runs offline from packaged example data.
- **[checkpoint_modifications.ipynb](checkpoint_modifications.ipynb)** — Editing
  a converted checkpoint's config as plain data — e.g. adding a surface-pressure
  output and a filter that fixes the global-mean surface pressure — with no gin.
- **[climate_stability.ipynb](climate_stability.ipynb)** — Driving the stable
  coarse models far past weather lead times (1.4° stochastic for ~6 months, 2.8°
  precipitation for ~2 years) with real ERA5 seasonal forcing, tracking global
  stability indicators, T850 snapshots and the zonal-mean jet.

## Datasets

The original NeuralGCM team's pre-computed simulation outputs (Zarr on Google
Cloud Storage) are catalogued in the upstream
[NeuralGCM simulation datasets](https://neuralgcm.readthedocs.io/en/latest/neuralgcm_datasets.html)
documentation.
