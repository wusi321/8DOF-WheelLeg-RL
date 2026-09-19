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

from ..stance import CROUCH_STANCE, FOLDED_STANCE, NOMINAL_STANCE, leg_joint_positions

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

  The standing share is the larger one: balance and travel are what the machine
  is for, and a command mix that spends most of its time on the ground starves
  the walking data the rest of the reward is written for.
  """

  folded_fraction: float = 0.30
  standing_fraction: float = 0.45
  transition_rate: float = 2.0
  """How fast the commanded posture may travel, in alpha per second."""

  def build(self, env) -> PostureCommand:
    return PostureCommand(self, env)


class PostureCommand(CommandTerm):
  cfg: PostureCommandCfg

  def __init__(self, cfg: PostureCommandCfg, env):
    super().__init__(cfg, env)
    self._env = env
    self.alpha = torch.zeros(self.num_envs, device=self.device)
    self._target = torch.zeros(self.num_envs, device=self.device)
    self._resample_command(torch.arange(self.num_envs, device=self.device))

  @property
  def command(self) -> torch.Tensor:
    """[B, 1] commanded posture, 0 folded to 1 standing."""
    return self.alpha.unsqueeze(-1)

  def _update_metrics(self) -> None:
    """Publish the commanded posture for logging.

    Note what this does *not* report: ``CommandTerm.reset`` averages a metric
    over the environments that reset at that step, so the logged value is a
    handful of environments, not the fleet. It swings between 0 and 1 for that
    reason alone and is not comparable across runs.
    """
    self.metrics["posture_alpha"] = self.alpha

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    draw = torch.rand(count, device=self.device)
    folded = draw < self.cfg.folded_fraction
    standing = draw >= 1.0 - self.cfg.standing_fraction
    value = torch.rand(count, device=self.device)
    value = torch.where(folded, torch.zeros_like(value), value)
    value = torch.where(standing, torch.ones_like(value), value)

    # The pose an environment was just spawned in decides its first command. The
    # event manager runs before this term inside ``_reset_idx``, so a spawn can
    # leave the posture its pose implies; without that coupling the two are drawn
    # independently and a standing spawn is handed a folded command about two
    # times in three, at which point the action offset drags its legs out from
    # under it on the very first step. That is a guaranteed face-plant, and it
    # taught the policy that standing is hopeless. The flag is consumed here so
    # later resamples mid-episode are free again.
    forced = getattr(self._env, "_spawn_posture", None)
    if forced is not None:
      override = forced[env_ids]
      value = torch.where(torch.isnan(override), value, override)
      forced[env_ids] = float("nan")
    self._target[env_ids] = value
    # Start *at* the spawned posture: a ramp from a stale value would move the
    # legs before the episode has begun.
    self.alpha[env_ids] = value

  def _update_command(self) -> None:
    """Ramp the commanded posture towards its target instead of jumping.

    A step change in alpha moves every leg target discontinuously, so the legs
    snap to the new pose and the robot topples -- which the policy cannot tell
    apart from "this task is impossible". Rate-limiting makes both directions a
    reachable trajectory it can keep balance through, and turns "lie down from
    standing" into a skill rather than a fall.
    """
    step = self.cfg.transition_rate * self._env.step_dt
    delta = torch.clamp(self._target - self.alpha, -step, step)
    self.alpha[:] = self.alpha + delta


def posture_alpha(env, command_name: str = POSTURE_COMMAND_NAME) -> torch.Tensor:
  """Commanded posture fraction [B], 0 folded to 1 standing."""
  return env.command_manager.get_term(command_name).alpha


def target_joint_pos(alpha: torch.Tensor, device=None) -> torch.Tensor:
  """[B, 6] leg joint targets for a commanded posture.

  Order is left hip, thigh, knee, right hip, thigh, knee, matching
  ``standing._LEG_JOINTS``. The expansion goes through ``leg_joint_positions``
  so the right hip is negated: the hip axes are not mirrored in the model.
  """
  where = device or alpha.device
  folded = torch.tensor([leg_joint_positions(FOLDED_STANCE)], device=where, dtype=alpha.dtype)
  standing = torch.tensor([leg_joint_positions(NOMINAL_STANCE)], device=where, dtype=alpha.dtype)
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
