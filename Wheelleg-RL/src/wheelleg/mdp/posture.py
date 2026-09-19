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
  """Samples a commanded body posture, held until it is resampled.

  Standing is the only command a running episode ever issues. The folded pose is
  a *reset* condition instead -- the spawn mix decides how many episodes begin on
  the ground -- because lying down is not a locomotion task: issuing it to a
  walking robot interrupts it and makes it fall on purpose, and moving while
  folded is the crawl this task exists to avoid.

  The half-crouch that used to sit between the two ends is gone as well. It was
  the arithmetic midpoint of the two stances, never solved for axle position, so
  its wheel sat forward of the body and rolling in it tipped the machine onto its
  back; both ends of the range are poses the geometry was actually solved for.
  """

  transition_rate: float = 1.0
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
    # Two values are involved and they are not interchangeable. ``command`` is
    # where the posture is told to go -- standing, always, except for a folded
    # spawn that was told to stay flat. ``start`` is where it begins, which only a
    # spawn knows, because the spawn is what just wrote the pose.
    #
    # A reset may set alpha straight to ``start`` precisely because the pose
    # already matches it. A timer tick must not: it may only move the target and
    # let ``_update_command`` ramp, or a robot holding the folded pose is snapped
    # upright in a single step, which is the one thing the ramp exists to prevent.
    command = torch.ones(len(env_ids), device=self.device)
    start = None
    pose = getattr(self._env, "_spawn_alpha", None)
    want = getattr(self._env, "_spawn_command", None)
    if pose is not None and want is not None:
      start = pose[env_ids]
      command = torch.where(torch.isnan(want[env_ids]), command, want[env_ids])
      # Consumed, so a mid-episode resample is a free draw again.
      pose[env_ids] = float("nan")
      want[env_ids] = float("nan")
    self._target[env_ids] = command
    if start is not None:
      reset = ~torch.isnan(start)
      self.alpha[env_ids] = torch.where(reset, start, self.alpha[env_ids])

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
