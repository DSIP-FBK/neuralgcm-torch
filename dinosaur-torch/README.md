# dinosaur-torch
<p align="center">
  <img src="https://raw.githubusercontent.com/DSIP-FBK/neuralgcm-torch/main/dinosaur-torch/docs/dinosaur-torch.png"
       width="240" alt="dinosaur-torch">
</p>

<p align="center">
  <a href="https://pypi.org/project/dinosaur-torch/"><img src="https://img.shields.io/pypi/v/dinosaur-torch?logo=pypi&logoColor=white" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <a href="https://github.com/DSIP-FBK/neuralgcm-torch"><img src="https://img.shields.io/badge/GitHub-DSIP--FBK%2Fneuralgcm--torch-181717?logo=github" alt="GitHub"></a>
  <a href="https://github.com/DSIP-FBK/neuralgcm-torch/actions/workflows/ci.yml"><img src="https://github.com/DSIP-FBK/neuralgcm-torch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License: Apache 2.0">
</p>

An idiomatic-PyTorch rewrite of [Dinosaur](https://github.com/neuralgcm/dinosaur),
the spectral dynamical core behind NeuralGCM. Rather than a line-for-line
JAX→PyTorch translation, this package is written the way a PyTorch library
would be written from scratch, while staying numerically equivalent to the
original.

Numerically validated against the original JAX Dinosaur.

## Design

- **Tensors in, tensors out.** Functions and modules operate on
  `torch.Tensor`s. NumPy appears only at I/O and construction boundaries
  (grid/quadrature setup, xarray conversion). There is no `asarray` promotion
  at every call site, no global default-device convention, and no
  host-constant device cache.
- **Precomputed constants live in `nn.Module` buffers.** Objects that hold
  tensors (spectral transforms, the dycore) are `torch.nn.Module`s with
  *non-persistent* buffers, so `.to(device)` / `.float()` work the standard
  way and `state_dict()` contains only learned parameters (none, for the
  dycore).
- **Static metadata is separate from tensors.** `GridSpec`,
  `SigmaCoordinates` etc. are frozen dataclasses — hashable, comparable,
  cheap — used to *construct* the tensor-holding modules.
- **States are torch pytrees.** Model state (`State`, diagnostics, …) is a
  plain dataclass registered via `torch.utils._pytree.register_dataclass`, so
  it composes natively with `torch.compile`, `torch.func`, and CUDA graphs.
  No custom pytree registry.
- **Standard test style:** plain `pytest` with parametrization (no
  absl/parameterized).
- **Scope:** the primitive-equations path used by NeuralGCM (transforms,
  sigma coordinates, primitive equations, IMEX time integration, filtering,
  vertical/horizontal interpolation, data utilities). Shallow-water and
  Held–Suarez model families are intentionally not ported: no published
  NeuralGCM checkpoint uses them.

## Layout

| module | contents |
|---|---|
| `associated_legendre.py`, `fourier.py` | basis construction (NumPy, at setup time) |
| `spherical_harmonic.py` | `GridSpec` (static), `RealSphericalHarmonics` / `FastSphericalHarmonics` transforms, `Grid` (`nn.Module`: transforms + spectral operators) |
| `sigma_coordinates.py` | `SigmaCoordinates` (static) + `SigmaLevels` (`nn.Module`: vertical finite-difference / integral operators) |
| `coordinate_systems.py` | `CoordinateSystem` (`nn.Module`: horizontal × vertical), spectral up/downsampling |
| `primitive_equations.py` | `State` (torch-pytree dataclass), `PrimitiveEquations` (`nn.Module` IMEX ODE, dry/moist/cloud variants), `Geopotential` |
| `time_integration.py` | IMEX Runge-Kutta steppers (SIL3, CN-RK2/3/4, Euler), step filters, trajectories (plain loops), digital filter initialization |
| `filtering.py` | exponential / horizontal-diffusion spectral filters |
| `vertical_interpolation.py` | `PressureCoordinates` / `PressureLevels`, pressure ↔ sigma regridding (batched searchsorted/gather, no vmap) |
| `horizontal_interpolation.py` | conservative / bilinear / nearest lat-lon regridders (weights precomputed as buffers) |
| `radiation.py` | top-of-atmosphere incident solar radiation (`SolarRadiation` module) |
| `scales.py`, `units.py` | unit handling / nondimensionalization (NumPy + pint) |
| `xarray_utils.py` | ERA5-style dataset preparation: `regrid_horizontal`, `fill_nan_with_nearest`, `selective_temporal_shift`, `grid_spec_from_dataset` |
| `pytree.py` | tiny helpers over `torch.utils._pytree` |

Both spherical-harmonics layouts are implemented because published NeuralGCM
checkpoints use both: `RealSphericalHarmonics` (modal shape `(2M-1, L)`, the
2.8° deterministic checkpoint) and `FastSphericalHarmonics` (zero-imag layout,
modal shape `(2M, L)`, e.g. the TL63 stochastic checkpoint; named
`RealSphericalHarmonicsWithZeroImag` upstream).

## Status

The dycore and data path are complete and numerically validated against the
original JAX implementation — transforms, operators, the full
primitive-equations step (dry and moist, including a 10-step baroclinic-wave
trajectory), vertical/horizontal regridding, and solar radiation all match to
1e-5–1e-4 of each field's range — alongside 141 unit tests (pytest). A full
SIL3 time step compiles with
`torch.compile(fullgraph=True)` out of the box — no shim rework, no graph
breaks.

> **Not ported:** shallow water, Held–Suarez, hybrid coordinates, and leapfrog
steppers (intentionally out of scope — no published NeuralGCM checkpoint
uses them).
