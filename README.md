<p align="center">
  <img src="https://raw.githubusercontent.com/DSIP-FBK/neuralgcm-torch/main/neuralgcm-torch/notebooks/neuralgcm-torch.png"
       width="440" alt="neuralgcm-torch">
</p>

<p align="center">
  <a href="https://github.com/DSIP-FBK/neuralgcm-torch/actions/workflows/ci.yml"><img src="https://github.com/DSIP-FBK/neuralgcm-torch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://dsip-fbk.github.io/neuralgcm-torch/"><img src="https://img.shields.io/badge/docs-Jupyter%20Book-F37726?logo=jupyter&logoColor=white" alt="Docs"></a>
  <a href="https://pypi.org/project/neuralgcm-torch/"><img src="https://img.shields.io/pypi/v/neuralgcm-torch?logo=pypi&logoColor=white&label=neuralgcm-torch" alt="neuralgcm-torch on PyPI"></a>
  <a href="https://pypi.org/project/dinosaur-torch/"><img src="https://img.shields.io/pypi/v/dinosaur-torch?logo=pypi&logoColor=white&label=dinosaur-torch" alt="dinosaur-torch on PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License: Apache 2.0"></a>
  <a href="https://www.nature.com/articles/s41586-024-07744-y"><img src="https://img.shields.io/badge/paper-Nature%202024-b31b1b" alt="Paper"></a>
</p>

<p align="center">
  <video autoplay loop muted playsinline width="480"
         poster="https://raw.githubusercontent.com/DSIP-FBK/neuralgcm-torch/main/neuralgcm-torch/notebooks/media/forecast_0_7_deg_poster.png">
    <source src="https://raw.githubusercontent.com/DSIP-FBK/neuralgcm-torch/main/neuralgcm-torch/notebooks/media/forecast_0_7_deg.mp4" type="video/mp4">
    <img src="https://raw.githubusercontent.com/DSIP-FBK/neuralgcm-torch/main/neuralgcm-torch/notebooks/media/forecast_0_7_deg.gif"
         width="480" alt="A 12-day NeuralGCM-0.7° forecast on a rotating globe">
  </video>
  <br>
  <em>A 12-day NeuralGCM-0.7° forecast — 850&nbsp;hPa specific humidity — on a slowly rotating globe.</em>
</p>

PyTorch implementations of [NeuralGCM](https://github.com/neuralgcm/neuralgcm)
and its spectral dynamical core,
[Dinosaur](https://github.com/neuralgcm/dinosaur) — idiomatic `nn.Module`
models that run the published NeuralGCM checkpoints with **no JAX, gin or
haiku at runtime**.

This repository contains two packages:

- **[neuralgcm-torch](neuralgcm-torch/)** — the NeuralGCM model (hybrid ML +
  physics atmospheric model): xarray-in/out inference, encode / advance /
  decode, ensembles, training, multi-GPU.
- **[dinosaur-torch](dinosaur-torch/)** — the standalone spectral dynamical
  core (spherical-harmonic transforms, sigma coordinates, primitive-equation
  IMEX time stepping).

## Install

```sh
pip install neuralgcm-torch        # pulls in dinosaur-torch
# checkpoint downloads also need: pip install "neuralgcm-torch[hub]"
```

## Quick start

```python
import neuralgcm_torch as neuralgcm
from neuralgcm_torch import pretrained

path = pretrained.fetch_checkpoint('deterministic_2_8_deg')  # cached Hub download
model = neuralgcm.PressureLevelModel.from_checkpoint(path, device='cuda')
```

See [neuralgcm-torch/README.md](neuralgcm-torch/README.md) and
[dinosaur-torch/README.md](dinosaur-torch/README.md) for details, and
[neuralgcm-torch/notebooks/](neuralgcm-torch/notebooks/) for end-to-end
forecast, ensemble and climate-stability examples.

## Documentation

The example notebooks — forecasting at every resolution, batched ensembles,
precipitation / evaporation, multi-decade climate stability, and the model
internals — are rendered with their outputs as a **Jupyter Book** at
**[dsip-fbk.github.io/neuralgcm-torch](https://dsip-fbk.github.io/neuralgcm-torch/)**.

## Development

```sh
uv sync                  # both packages editable + dev tools
pre-commit install
DINOSAUR_TORCH_TEST_DEVICE=cpu uv run pytest dinosaur-torch
```

## License

Apache 2.0 (code) — see [LICENSE](LICENSE) and [NOTICE](NOTICE). These packages
are a PyTorch port of Google's NeuralGCM and Dinosaur. The converted model
checkpoints, hosted on the Hugging Face Hub, are © Google LLC and licensed
under CC BY-SA 4.0 (see the model card).
