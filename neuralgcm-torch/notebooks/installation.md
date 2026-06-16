# Installation

`neuralgcm-torch` needs Python ≥ 3.11 and PyTorch. A CUDA GPU is strongly
recommended — the higher-resolution and climate-stability runs assume one.

You can grab any notebook from this site with the **download button** at the
top-right of its page; all you then need is an environment with the package
installed. Two ways to get one.

## Install from PyPI

A fresh virtual environment with [uv](https://docs.astral.sh/uv/):

```sh
uv venv
uv pip install 'neuralgcm-torch[hub,notebooks]'
```

…or with `pip`:

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install 'neuralgcm-torch[hub,notebooks]'
```

This pulls in `dinosaur-torch` (the dynamical core) as well. The `[hub]` extra
adds Hugging Face support so `pretrained.fetch_checkpoint` can download the
converted checkpoints (cached on first use); the `[notebooks]` extra adds what
the example notebooks need on top of the package — `matplotlib` for plots and
`gcsfs`/`zarr` to read the public ERA5 archive. For the package alone, plain
`'neuralgcm-torch[hub]'` is enough.

```python
import neuralgcm_torch as neuralgcm
from neuralgcm_torch import pretrained

path = pretrained.fetch_checkpoint('deterministic_2_8_deg')  # cached Hub download
model = neuralgcm.PressureLevelModel.from_checkpoint(path, device='cuda')
```

## Clone the repository (development)

To work on the code, clone the repo and let `uv` set up the whole workspace —
both packages editable, plus the dev and notebook tooling — in one step:

```sh
git clone https://github.com/DSIP-FBK/neuralgcm-torch
cd neuralgcm-torch
uv sync
```

Then launch Jupyter from the repository root:

```sh
uv run --with jupyterlab jupyter lab
```
