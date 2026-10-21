# Third-party notices

This repository contains one vendored third-party tree and is informed by several other
open-source projects. Nothing else in the repository is copied from third-party code: the
rest of the source (`Wheelleg-RL/src/`, `Wheelleg-RL/scripts/`, `Wheelleg-RL/tests/`,
`robot_description/`, `mechanical/`) is this project's own work and is MIT licensed.

## Vendored code

### mjlab — `Wheelleg-RL/mjlab/`

- Upstream: <https://github.com/mujocolab/mjlab>
- License: **Apache License 2.0** (full text kept at `Wheelleg-RL/mjlab/LICENSE`)
- Nature: a pinned, locally patched snapshot of the mjlab framework, not project code.
  It is vendored because the training task depends on a specific unreleased state of the
  framework, and a floating dependency made training runs non-reproducible.
- Provenance, the list of local modifications, and how to re-vendor:
  [`Wheelleg-RL/mjlab/PROVENANCE.md`](Wheelleg-RL/mjlab/PROVENANCE.md)
- Upstream copyright and attribution notices are preserved unmodified. The Apache-2.0
  license of this directory is not superseded by this project's MIT license.

## Referenced projects (no code vendored)

### RC_WheelLeg — <https://github.com/zeitvex/RC_WheelLeg>

- License: MIT
- Relationship: the 16-DOF competition platform this 8-DOF training project precedes. Its
  project layout, deployment boundary (Python and ROS 2 / C++ runtimes) and field results
  informed the design here. The robot description, MJCF, task code, rewards and curricula
  in this repository were written for the 8-DOF machine; shared engineering experience is
  attributed rather than copied.

### MicroDuck RL — <https://github.com/pollen-robotics/microduck_rl>

- License: see upstream repository
- Relationship: the fall-recovery structure (potential-based progress rewards, no positive
  reward gated on a bad state, a finite recovery window, reverse-curriculum prone spawns)
  follows the published lessons from MicroDuck's VelStand task. The implementation in
  `Wheelleg-RL/src/wheelleg/mdp/` targets this robot's geometry and is this project's own
  code; the design rationale is cited in `Wheelleg-RL/README.md`.

### 16-DOF reference training project

- An internal HYNova 16-DOF training project was consulted as a reference for actuator
  scaling, action scaling and low-pass filters. **It is not published in this repository
  and is not a runtime dependency**; it is referenced in the documentation only, by name,
  where a number was derived from it (notably the placeholder torque limits).

## Runtime dependencies

Installed from package indexes, not vendored: MuJoCo and MuJoCo Warp, PyTorch, rsl-rl,
`uv`, and the packages declared in `Wheelleg-RL/pyproject.toml` and
`Wheelleg-RL/uv.lock`. Each retains its own license; `uv.lock` is the exact resolution used
for the training runs recorded in the documentation.

## Media and models

The STL meshes under `robot_description/` and `Wheelleg-RL/mjcf/meshes/` are this project's
own robot geometry, exported from the SolidWorks sources in `mechanical/`. The mjlab
documentation images under `Wheelleg-RL/mjlab/docs/` belong to the mjlab project and stay
under its license.
