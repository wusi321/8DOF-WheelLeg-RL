# Contributing to 8DOF WheelLeg RL

Thanks for taking a look. This is a small, single-maintainer project, so the process is
deliberately short. The goal is that any change can be checked without a GPU, and that a
change to the reward, the curriculum or the model is reproducible from the commit message
and the tag it lands on.

（中文：这是一个单人维护的小工程，流程尽量短。改奖励、课程或模型时，请附一次 64 环境 /
5 iteration 的 smoke test 结果，并说明改前改后的实测数值。）

## Reporting a bug

Open an [issue](https://github.com/wusi321/8DOF-WheelLeg-RL/issues/new/choose) and include:

- what you ran (stage: `flat` / `rough` / `recovery`, `NUM_ENVS`, `ITERS`, GPU);
- the output of `uv run list-envs | grep Wheelleg`;
- the failing log lines, including the first traceback rather than the last;
- whether the run had a checkpoint loaded (`play`) or started from scratch (`train`).

For anything that looks like a numerical blow-up, say whether `NaN` appeared in the
reward or in the observations — the project has a guard for this and the distinction
changes the diagnosis.

## Making a change

1. Fork, branch from `main`, keep the change to one idea.
2. Run the dependency-free contract tests before opening the PR:

   ```bash
   cd Wheelleg-RL
   python -m unittest -v tests.test_stance tests.test_urdf_geometry tests.test_entity_bindings
   ```

   These need no GPU, no MuJoCo and no PyTorch, and they are what CI runs.
3. If you touched the robot model, re-run the converter and commit the regenerated MJCF:

   ```bash
   bash scripts/convert_model.sh
   ```

4. If you touched rewards, curricula or terrain, also report one smoke run:

   ```bash
   ITERS=5 NUM_ENVS=64 bash scripts/train.sh <stage>
   ```

5. Update the relevant section of [`Wheelleg-RL/README.md`](Wheelleg-RL/README.md) when the
   behaviour or the numbers change. That document is the design record; a reward change
   without a recorded measurement is hard to review.

## Commit messages

One line, present tense, saying what the change does *and why it matters*. Compare:

- `update rewards` — not reviewable;
- `Stop taxing the leg motion that climbing requires` — says what and why.

If the change fixes a regression, name the symptom in the message (`Index the fall test by
env_ids: it crashed the curriculum`). Tags are cut from `main` and each tag should point at
a training configuration that reproduces a working policy.

## Scope

Pull requests for the training stack, the robot description and the documentation are
welcome. Real-machine drivers, ROS 2 deployment and standalone ONNX Sim2Sim tooling are
out of scope for this repository (they live with the 16-DOF platform,
[RC_WheelLeg](https://github.com/zeitvex/RC_WheelLeg)).

## Code style

- Python 3.10+, type hints where they pay for themselves, `ruff` clean
  (`uv run ruff check src scripts tests`).
- Do not commit `logs/`, `wandb/`, checkpoints, API keys or W&B entity secrets.
- Do not vendor new third-party trees. If a dependency needs a patch, document it in
  `THIRD_PARTY_NOTICES.md` instead of copying the package in.
