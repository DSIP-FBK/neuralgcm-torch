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
"""Fetch converted checkpoints from the Hugging Face Hub.

The published NeuralGCM checkpoints, converted to the -next format, are
hosted on the Hub so they load with no legacy package, no GCS access and
no conversion step:

    from neuralgcm_torch import PressureLevelModel, pretrained
    path = pretrained.fetch_checkpoint('deterministic_2_8_deg')
    model = PressureLevelModel.from_checkpoint(path, device='cuda')

`huggingface_hub` is an optional dependency (``pip install
'neuralgcm-torch[hub]'``); only `fetch_checkpoint` needs it.

Hosting/licensing note: the converted weights are derivative works of
Google's NeuralGCM v1 checkpoints, which are CC BY-SA 4.0, so the Hub
repo carries that license and attribution.
The conversion *code* in this package is Apache 2.0 — different licenses
for code vs. weights, which is exactly why the weights live on the Hub
and the code on GitHub.
"""

from __future__ import annotations

import os

# Default Hub repo id; override with the NEURALGCM_TORCH_HF_REPO env var.
DEFAULT_REPO_ID = os.environ.get(
    'NEURALGCM_TORCH_HF_REPO', 'it4lia/neuralgcm-torch'
)

# name -> (resolution label, parameter count, one-line description); the
# single source of truth for both `fetch_checkpoint` and the model card.
CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    'deterministic_0_7_deg': ('0.7° (TL255)', '31.1M', 'deterministic'),
    'deterministic_1_4_deg': ('1.4° (TL127)', '18.3M', 'deterministic'),
    'deterministic_2_8_deg': ('2.8° (TL63)', '14.5M', 'deterministic'),
    'stochastic_1_4_deg': ('1.4° (TL127)', '11.5M', 'stochastic (NeuralGCM-ENS)'),
    'stochastic_precip_2_8_deg': ('2.8° (TL63)', '11.1M', 'stochastic, precipitation'),
    'stochastic_evap_2_8_deg': ('2.8° (TL63)', '11.1M', 'stochastic, evaporation'),
    'tl63_stochastic_mini': ('TL63 toy', '0.19M', 'stochastic toy / test fixture'),
}


def fetch_checkpoint(
    name: str,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    local_root: str | None = None,
) -> str:
  """Returns a local path to a converted checkpoint.

  Resolves **local-first**: if the file already exists under `local_root`
  (or the ``NEURALGCM_TORCH_CHECKPOINTS`` directory), that path is returned
  and nothing is downloaded — so this works offline and before anything is
  published. Otherwise the file is downloaded from the Hub (cached, so
  repeated calls are cheap). Pass the returned path to
  `PressureLevelModel.from_checkpoint`.

  Args:
    name: checkpoint name, with or without the ``.pt`` suffix (e.g.
      ``'deterministic_2_8_deg'``); see `CHECKPOINTS` for the published set.
    repo_id: Hub repo id; defaults to `DEFAULT_REPO_ID` (overridable via the
      ``NEURALGCM_TORCH_HF_REPO`` environment variable).
    revision: optional Hub revision (branch / tag / commit).
    cache_dir: optional Hub cache directory.
    local_dir: if given, the downloaded file is placed there instead of the cache.
    local_root: a directory of already-converted ``.pt`` files to reuse before
      downloading (also read from the ``NEURALGCM_TORCH_CHECKPOINTS`` env var).
  """
  filename = name if name.endswith('.pt') else f'{name}.pt'
  for root in (local_root, os.environ.get('NEURALGCM_TORCH_CHECKPOINTS')):
    if root:
      candidate = os.path.join(root, filename)
      if os.path.exists(candidate):
        return candidate
  try:
    from huggingface_hub import hf_hub_download
  except ImportError as e:  # pragma: no cover - import-guard
    raise ImportError(
        "fetch_checkpoint needs huggingface_hub to download "
        f"{filename!r} (no local copy found); install it with "
        "`pip install 'neuralgcm-torch[hub]'`."
    ) from e
  return hf_hub_download(
      repo_id=repo_id or DEFAULT_REPO_ID,
      filename=filename,
      revision=revision,
      cache_dir=cache_dir,
      local_dir=local_dir,
  )
