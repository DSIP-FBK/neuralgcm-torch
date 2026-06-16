# Copyright 2024 Google LLC
# Copyright 2026 Fondazione Bruno Kessler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modules that advance the state with the dycore plus physics tendencies.

A corrector is called as `corrector(state, tendencies, forcing=None)` where
`state` is a modal `primitive_equations.State` and `tendencies` holds
(possibly `None`-leaved) physics tendencies treated as constant over the
step. Only the correctors referenced by published checkpoint configs are
ported.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from dinosaur_torch import coordinate_systems
from dinosaur_torch import time_integration
from neuralgcm_torch import filters


class DycoreWithPhysicsCorrector(nn.Module):
  """Runs the dycore with physics tendencies added to the explicit terms.

  The physics tendencies are held constant over all `substeps` dycore steps
  of size `dt / substeps`; the step filter is applied after each substep.

  Args:
    equation: the dycore `ImplicitExplicitODE` (a
      `primitive_equations.PrimitiveEquations` module).
    dt: the corrector's (outer) nondimensional time step.
    substeps: number of inner dycore steps per call.
    time_integrator: IMEX integrator, e.g. `time_integration.imex_rk_sil3`.
    filter_module: step filter applied after each substep (built with the
      outer `dt`, matching the legacy behavior).
    orography_module: optional module producing the modal orography used by
      `equation`; held here so `import_haiku` can refresh the equation's
      orography buffer after loading learned corrections.
  """

  HAIKU_NAME = 'dycore_with_physics_corrector'

  def __init__(
      self,
      equation: time_integration.ImplicitExplicitODE,
      dt: float,
      substeps: int,
      time_integrator: Callable = time_integration.imex_rk_sil3,
      filter_module: Optional[nn.Module] = None,
      orography_module: Optional[nn.Module] = None,
  ):
    super().__init__()
    self.equation = equation
    self.dt = dt
    self.substeps = substeps
    self.inner_dt = dt / substeps
    self.time_integrator = time_integrator
    self.filter = filter_module or filters.NoFilter()
    self.orography_module = orography_module

  def forward(self, state, tendencies, forcing=None):
    del forcing  # unused
    physics_eq = time_integration.ExplicitODE.from_functions(
        lambda _: tendencies
    )
    equation = time_integration.compose_equations(
        [self.equation, physics_eq]
    )
    step_fn = self.time_integrator(equation, self.inner_dt)
    step_fn = time_integration.step_with_filters(step_fn, [self.filter])
    step_fn = time_integration.repeated(step_fn, self.substeps)
    return time_integration.maybe_fix_sim_time_roundoff(
        step_fn(state), self.dt
    )

  def import_haiku(self, params: dict, prefix: str) -> None:
    # the legacy equation constructor creates the orography module directly
    # in the corrector's scope (the equation itself is not a haiku module).
    if self.orography_module is not None and hasattr(
        self.orography_module, 'import_haiku'
    ):
      self.orography_module.import_haiku(
          params, f'{prefix}/~/{self.orography_module.HAIKU_NAME}'
      )
      # the equation snapshots the modal orography as a buffer; refresh it
      # now that the learned correction parameters are loaded. The snapshot
      # is grad-free: during training the dycore sees frozen orography
      # (orography corrections still receive gradients through the
      # encoder/decoder paths, which call the module in their forward).
      with torch.no_grad():
        self.equation.orography.copy_(
            self.orography_module().to(self.equation.orography.dtype)
        )


class CustomCoordsCorrector(nn.Module):
  """Corrector that runs an inner corrector on different coordinates.

  The modal state and tendencies are spectrally interpolated to
  `custom_coords` (the inner corrector's coordinate system), advanced, and
  interpolated back.
  """

  HAIKU_NAME = 'custom_coords_corrector'

  def __init__(
      self,
      coords: coordinate_systems.CoordinateSystem,
      custom_coords: coordinate_systems.CoordinateSystem,
      corrector: nn.Module,
  ):
    super().__init__()
    self.corrector = corrector
    self.to_custom_coords_fn = coordinate_systems.get_spectral_interpolate_fn(
        coords, custom_coords
    )
    self.from_custom_coords_fn = (
        coordinate_systems.get_spectral_interpolate_fn(custom_coords, coords)
    )

  def forward(self, state, tendencies, forcing=None):
    del forcing  # not supported (matches the legacy corrector)
    state = self.to_custom_coords_fn(state)
    tendencies = self.to_custom_coords_fn(tendencies)
    custom_out = self.corrector(state, tendencies, None)
    return self.from_custom_coords_fn(custom_out)

  def import_haiku(self, params: dict, prefix: str) -> None:
    self.corrector.import_haiku(
        params, f'{prefix}/~/{self.corrector.HAIKU_NAME}'
    )
