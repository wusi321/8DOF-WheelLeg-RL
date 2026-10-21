# Changelog

All notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions before `1.0.0` are
tagged training snapshots — each tag freezes one configuration that reproduces a working
policy, which is how a curriculum change that regresses the robot gets rolled back to a
known-good run.

## [1.0.0] - 2026-10-22

First documented release. Consolidates the 48 commits made between 2026-09-17 and
2026-10-09 (tags `v0.1.0` … `v0.4.15`) into one reproducible training stack.

### Added

- URDF/Xacro → MJCF converter (`scripts/urdf_to_mjcf.py`) that normalises the joint names
  exported by SolidWorks, recentres the wheel mesh, separates visual from collision
  geometry, and produces the model the training task actually loads.
- Three registered tasks: `Wheelleg-Flat-v0`, `Wheelleg-Rough-v0`, `Wheelleg-Recovery-v0`.
- Staged curriculum: balance → velocity tracking (including lateral stepping) → slopes,
  Perlin/random rough terrain, low walls and two stair variants (obstacle height capped at
  12 cm) → fall recovery and commanded crouch/prone postures.
- Domain randomisation for push disturbances, friction, base mass, actuator gains and
  action latency, plus an optional height-scanner / raycast critic.
- Fall-recovery economics modelled on MicroDuck's VelStand task: potential-based progress
  rewards, a one-shot recovery bounty, and a hysteresis-gated fallen penalty — deliberately
  no positive reward gated on a bad state, which is what makes "lie still" a losing policy.
- Reverse curriculum (`spawn_fallen`): 50 % of episodes start standing, 35 % fully folded,
  15 % in a mid-recovery crouch, so standing up is actually explored.
- Contract tests that need no GPU, no MuJoCo and no PyTorch: stance geometry and clearance
  thresholds, URDF→MJCF reproducibility, wheel-mesh concentricity, MDP-term/entity binding
  (60 tests), plus three simulator-level tests for the model contract, standing numerics
  and wheel tracking.
- CI (`validate` workflow) running the dependency-free tests on every push and pull request.
- `CONTRIBUTING.md`, issue forms, a pull-request template, `CITATION.cff`,
  `THIRD_PARTY_NOTICES.md` and a provenance record for the vendored mjlab snapshot.

### Changed

- Posture and ground contact are reward signals, not termination conditions: a robot that
  kneels is charged per second instead of being reset before it can experience the
  consequence, and `fallen_too_long` (2.5 s) replaces the old immediate reset on tilt.
- Leg symmetry is gated on measured wheel-height difference and on lateral commands, so the
  body can stay level on a step or a side slope instead of leaning with the terrain.
- Terrain curriculum advancement is judged on path length over complete, upright episodes
  rather than net displacement, which previously demoted every turning robot each episode
  and pinned the whole run to level 0.
- Commanded posture changes ramp at 2.0 /s instead of stepping, and the spawn pose is
  coupled to the first posture command so the legs are not yanked on the first frame.

### Fixed

- Wheel contact sensor pattern, robot entity naming and inertia regularisation for MuJoCo.
- Floating base and contact constraints; base mass randomisation kept physically consistent.
- Two-wheel velocity tracking reward; wheel meshes centred with a documented 2.546 mm
  local-Z offset.
- Mirror convention for the hip joints (mirror symmetry is the *sum* for hips, the
  *difference* for thigh/knee), single-sourced in `stance.leg_joint_positions()`.
- Fall-test indexing by `env_ids` (it crashed the curriculum) and height-error
  normalisation by a constant instead of a target that reaches zero.
- Removed positive-reward clipping, which made crawling penalties unable to go negative.

### Known limitations

- Actuator torque limits (legs ±4 N·m, wheels ±2 N·m) are placeholders scaled from the
  16-DOF reference machine; the source URDF carries no effort data. Replacing them with
  real motor figures only requires editing `src/wheelleg/actuator_spec.py` and re-running
  the converter.
- `Wheelleg-Recovery-v0` is still a short-episode variant of the rough recipe; it is not a
  complete recovery curriculum yet.
- No real-machine driver, ROS 2 deployment or standalone ONNX Sim2Sim tool is included.

## Earlier tagged snapshots

`v0.1.0` … `v0.4.15` (2026-09-17 → 2026-10-09) are unreleased snapshots kept for
reproducibility; `git log <tag>` shows exactly what each one changed. Notable steps:

| Tag | Change |
|---|---|
| `v0.2.2` | MIT license notice |
| `v0.3.0` | Actuator torque limits single-sourced, static demand reported |
| `v0.3.1` | Posture became a reward signal instead of a reset |
| `v0.4.0` | Kneeling and wheel lifting forbidden, wheeled stance rewarded, leg symmetry added |
| `v0.4.2` | Fall-recovery economics modelled on the MicroDuck VelStand task |
| `v0.4.3` | Reverse-curriculum prone spawns |
| `v0.4.5` | Commanded posture, leg action follows it |
| `v0.4.11` | Recovery bounty armed for a prone start |
| `v0.4.13` | Terrain undiluted, chassis allowed to rest on a step |
| `v0.4.15` | Knee parameters tuned, last snapshot before this release |
