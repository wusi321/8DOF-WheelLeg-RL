## What changed

<!-- One paragraph. What does the robot do differently, and why does it matter? -->

## Checks

- [ ] `python -m unittest tests.test_stance tests.test_urdf_geometry tests.test_entity_bindings` passes (no GPU needed)
- [ ] If the robot model changed: the MJCF was regenerated with `bash scripts/convert_model.sh`
- [ ] If rewards, curriculum or terrain changed: a `ITERS=5 NUM_ENVS=64` smoke run was made and its result is in the description
- [ ] Documentation of the affected design decision was updated in `Wheelleg-RL/README.md`
- [ ] No `logs/`, `wandb/`, checkpoints, API keys or W&B secrets are included
