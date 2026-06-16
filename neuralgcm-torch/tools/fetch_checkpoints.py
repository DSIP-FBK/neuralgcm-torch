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

"""Download converted checkpoints from the Hub into a local directory.

Run once to populate `neuralgcm-torch/notebooks/checkpoints/` so the
notebooks load instantly (no GCS download + conversion):

  uv run --no-sync python neuralgcm-torch/tools/fetch_checkpoints.py

Pass names to fetch a subset, and `--repo-id` / `--out` to override the
Hub repo or destination. Needs the optional `huggingface_hub` dependency
(`pip install 'neuralgcm-torch[hub]'`).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

from neuralgcm_torch import pretrained


def main():
  default_out = pathlib.Path(__file__).resolve().parent.parent / 'notebooks' / 'checkpoints'
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      'names', nargs='*',
      help='checkpoint names to fetch (default: all published ones)')
  parser.add_argument('--repo-id', default=None,
                      help=f'Hub repo id (default: {pretrained.DEFAULT_REPO_ID})')
  parser.add_argument('--out', default=str(default_out),
                      help='destination directory')
  parser.add_argument('--symlink', action='store_true',
                      help='symlink from the Hub cache instead of copying')
  args = parser.parse_args()

  names = args.names or list(pretrained.CHECKPOINTS)
  unknown = [n for n in names if n not in pretrained.CHECKPOINTS]
  if unknown:
    parser.error(f'unknown checkpoints {unknown}; '
                 f'choose from {list(pretrained.CHECKPOINTS)}')

  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  for name in names:
    cached = pathlib.Path(pretrained.fetch_checkpoint(name, repo_id=args.repo_id))
    dst = out / f'{name}.pt'
    if dst.exists() or dst.is_symlink():
      dst.unlink()
    if args.symlink:
      dst.symlink_to(cached)
    else:
      shutil.copy(cached, dst)
    print(f'{name}: {dst}')


if __name__ == '__main__':
  main()
