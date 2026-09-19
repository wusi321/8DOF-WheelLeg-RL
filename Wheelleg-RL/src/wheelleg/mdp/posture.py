"""Commanded body posture: lying folded, standing, or anywhere between.

Why this exists
---------------
The leg action is a *delta* from the standing stance: ``target = default_offset +
raw_action * scale``, with the offset at the nominal stance and a scale of
0.125 rad for the hip and 0.25 rad for the thigh and knee. Commanding the folded
pose from there needs raw actions of

    hip +7.26    thigh +1.85    knee -5.55        (folded)
    hip +3.63    thigh +0.93    knee -2.77        (halfway up)

against an initial policy action std of 0.80, i.e. six to nine sigma. The policy
therefore cannot choose to fold: a folded spawn is real, but the actuators drag
the legs back to the stance within about 0.1 s and the robot can never return.

This command fixes that at the level where the problem lives. It generates a
posture fraction ``alpha`` per environment (0 = fully folded, 1 = standing) and
the position action's offset follows it, so *zero action means the commanded
posture*. Both directions then need only small corrections: the robot does not
have to discover a nine sigma action to lie down, nor to get up.

``alpha`` is observed, which matters: the actor does not observe its own height,
so a posture that depended on state rather than on a command would be invisible
to the policy and its action mapping unpredictable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ..stance import CROUCH_STANCE, FOLDED_STANCE, NOMINAL_STANCE

POSTURE_COMMAND_NAME = "posture"

# alpha = 0 is the folded pose, alpha = 1 the standing stance. The crouch pose is
# carried only for the spawn mix, not as a command point.
FOLDED_ALPHA = 0.0
STANDING_ALPHA = 1.0


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
  """Samples a commanded body posture, held for the resampling interval.

  ``folded_fraction`` and ``standing_fraction`` force that share of environments
  to the two extremes, because sampling alpha uniformly almost never produces
  either end and both are the states that actually have to be learned. The rest
  are uniform in between, which is the richer part of the curriculum: the way up
  from lying flat is a path the policy has to discover.
  """

  folded_fraction: float = 0.35
  standing_fraction: float = 0.35

  def build(self, env) -> PostureCommand:
    return PostureCommand(self, env)


class PostureCommand(CommandTerm):
  cfg: PostureCommandCfg

  def __init__(self, cfg: PostureCommandCfg, env):
    super().__init__(cfg, env)
    self.alpha = torch.zeros(self.num_envs, device=self.device)
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  @property
  def command(self) -> torch.Tensor:
    """[B, 1] commanded posture, 0 folded to 1 standing."""
    return self.alpha.unsqueeze(-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    draw = torch.rand(count, device=self.device)
    folded = draw < self.cfg.folded_fraction
    standing = draw >= 1.0 - self.cfg.standing_fraction
    value = torch.rand(count, device=self.device)
    value = torch.where(folded, torch.zeros_like(value), value)
    value = torch.where(standing, torch.ones_like(value), value)
    self.alpha[env_ids] = value

  def _update_command(self) -> None:
    """Nothing to integrate: the posture is held until it is resampled."""


def posture_alpha(env, command_name: str = POSTURE_COMMAND_NAME) -> torch.Tensor:
  """Commanded posture fraction [B], 0 folded to 1 standing."""
  return env.command_manager.get_term(command_name).alpha


def target_joint_pos(alpha: torch.Tensor, device=None) -> torch.Tensor:
  """[B, 6] leg joint targets for a commanded posture.

  Order is left hip, thigh, knee, right hip, thigh, knee, matching
  ``standing._LEG_JOINTS``.
  """
  folded = torch.tensor([FOLDED_STANCE * 2], device=device or alpha.device, dtype=alpha.dtype)
  standing = torch.tensor([NOMINAL_STANCE * 2], device=device or alpha.device, dtype=alpha.dtype)
  blend = alpha.unsqueeze(-1)
  return folded + (standing - folded) * blend


def posture_pose_error(env, command_name: str = POSTURE_COMMAND_NAME) -> torch.Tensor:
  """Mean squared leg joint deviation from the *commanded* posture.

  Replaces a fixed pull towards the standing stance: a folded-commanded robot
  must be allowed, and required, to stay folded.
  """
  asset = env.scene["wheelleg"]
  from .standing import _LEG_JOINTS  # Local: standing imports this module's peers.

  ids, _ = asset.find_joints(_LEG_JOINTS, preserve_order=True)
  joints = asset.data.joint_pos[:, ids]
  target = target_joint_pos(posture_alpha(env, command_name))
  return torch.mean(torch.square(joints - target), dim=1)


def posture_height_target(
    env, standing_height: torch.Tensor, command_name: str = POSTURE_COMMAND_NAME
) -> torch.Tensor:
  """Clearance the commanded posture should sit at, [B].

  Blends from the folded body's rest height to whatever standing clearance the
  caller wants (which already carries the at-speed crouch allowance).
  """
  from ..stance import FOLDED_REST_Z

  alpha = posture_alpha(env, command_name)
  return FOLDED_REST_Z + (standing_height - FOLDED_REST_Z) * alpha
