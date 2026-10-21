# Provenance of this vendored mjlab snapshot

This directory is a **pinned copy of [mjlab](https://github.com/mujocolab/mjlab)**, not code
written by this project. It is kept in-tree because the training task depends on a specific
state of the framework (its manager API, terrain generator and actuator models), and a
floating dependency silently changed training behaviour between runs.

## Source

| | |
|---|---|
| Upstream | <https://github.com/mujocolab/mjlab> |
| Declared version | `1.4.0` (`pyproject.toml` in this directory) |
| Base for comparison | tags `v1.4.0` and `v1.5.0` of the upstream repository |
| License | Apache License 2.0 — full text in `LICENSE`, kept unmodified |
| Copyright | mujocolab and mjlab contributors |

## What differs from upstream

Verified on 2026-10-22 by comparing git blob hashes of this directory (511 files) against
the upstream tag trees:

- **2 files added** (present upstream only after `v1.4.0`):
  `scripts/tools/terrain_explorer.py`, `tests/test_builtin_dcmotor_actuator.py`.
- **19 files already at their post-`v1.4.0`, `v1.5.0` state** — the snapshot sits between
  the two releases rather than exactly on `v1.4.0`.
- **7 files modified locally**, i.e. they match neither `v1.4.0` nor `v1.5.0`:

  ```text
  .github/workflows/ci.yml
  .github/workflows/release.yml
  docs/source/changelog.rst
  pyproject.toml
  src/mjlab/envs/mdp/dr/actuator.py
  src/mjlab/utils/buffers/circular_buffer.py
  uv.lock
  ```

  `pyproject.toml` and `uv.lock` carry this project's index and dependency overrides; the
  rest are small behavioural or documentation adjustments. Diff a file against upstream
  before relying on it:

  ```bash
  git clone --filter=blob:none --no-checkout https://github.com/mujocolab/mjlab /tmp/mjlab
  git -C /tmp/mjlab fetch --depth 1 origin tag v1.4.0
  git -C /tmp/mjlab show v1.4.0:src/mjlab/utils/buffers/circular_buffer.py \
    | diff - src/mjlab/utils/buffers/circular_buffer.py
  ```

- No files were removed.

## Updating

1. Re-vendor the desired upstream tag over this directory, keeping `LICENSE` and this file.
2. Re-apply the local patches listed above (check each one still applies).
3. Regenerate `uv.lock` (`uv lock`) and commit it with the snapshot.
4. Run the contract tests (`python -m unittest tests.test_stance`) and one 5-iteration smoke
   training run before tagging a release that includes the update.

## Licensing

Apache-2.0 obligations follow this directory: the license text and upstream notices stay in
place, and `THIRD_PARTY_NOTICES.md` at the repository root records the vendoring. The MIT
license of the rest of the repository does not extend over this directory.
