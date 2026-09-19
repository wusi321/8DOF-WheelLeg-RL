"""Low-pass filtered action terms for mjlab (native implementation).

Provides FIR low-pass filtering on the raw policy output before applying
scale/offset. This smooths control signals and reduces sim-to-real gap.

- Leg joints:  5 Hz cutoff (slow, smooth)
- Wheel joints: 15 Hz cutoff (faster response needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import (
    JointPositionAction,
    JointPositionActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _lowpass_weights(control_freq: float, cutoff_freq: float) -> list[float]:
    """Compute 1st-order IIR low-pass filter weights.

    alpha = 1 - exp(-2蟺 * f_c / f_s)
    filtered = alpha * current + (1 - alpha) * previous
    """
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_freq / control_freq)
    return [alpha, 1.0 - alpha]


# -- Position low-pass --


@dataclass(kw_only=True)
class JointPositionLowPassActionCfg(JointPositionActionCfg):
    """Joint position action with 1st-order low-pass filter on policy output."""

    control_frequency: float = 50.0
    """Control frequency in Hz (= 1 / (dt * decimation))."""
    cut_off_frequency: float = 5.0
    """Cut-off frequency in Hz for the low-pass filter."""

    def build(self, env: ManagerBasedRlEnv) -> JointPositionLowPassAction:
        return JointPositionLowPassAction(self, env)


class JointPositionLowPassAction(JointPositionAction):
    """Applies a 1st-order low-pass filter to raw actions before processing."""

    def __init__(self, cfg: JointPositionLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._weights = _lowpass_weights(cfg.control_frequency, cfg.cut_off_frequency)
        self._prev_raw = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        filtered = self._weights[0] * actions + self._weights[1] * self._prev_raw
        self._prev_raw[:] = actions
        super().process_actions(filtered)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._prev_raw[env_ids] = 0.0
        super().reset(env_ids)


# -- Velocity low-pass --


@dataclass(kw_only=True)
class JointVelocityLowPassActionCfg(JointVelocityActionCfg):
    """Joint velocity action with 1st-order low-pass filter on policy output."""

    control_frequency: float = 50.0
    """Control frequency in Hz (= 1 / (dt * decimation))."""
    cut_off_frequency: float = 15.0
    """Cut-off frequency in Hz for the low-pass filter."""

    def build(self, env: ManagerBasedRlEnv) -> JointVelocityLowPassAction:
        return JointVelocityLowPassAction(self, env)


class JointVelocityLowPassAction(JointVelocityAction):
    """Applies a 1st-order low-pass filter to raw actions before processing."""

    def __init__(self, cfg: JointVelocityLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._weights = _lowpass_weights(cfg.control_frequency, cfg.cut_off_frequency)
        self._prev_raw = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        filtered = self._weights[0] * actions + self._weights[1] * self._prev_raw
        self._prev_raw[:] = actions
        super().process_actions(filtered)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._prev_raw[env_ids] = 0.0
        super().reset(env_ids)


@dataclass(kw_only=True)
class JointPositionDelayedLowPassActionCfg(JointPositionLowPassActionCfg):
    """Joint position action with random delay and 1st-order low-pass filter."""

    min_delay: int = 0
    max_delay: int = 2

    def build(self, env: ManagerBasedRlEnv) -> JointPositionDelayedLowPassAction:
        return JointPositionDelayedLowPassAction(self, env)


class JointPositionDelayedLowPassAction(JointPositionLowPassAction):
    """Applies a random delay and a 1st-order low-pass filter to raw actions."""

    def __init__(self, cfg: JointPositionDelayedLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.min_delay = cfg.min_delay
        self.max_delay = cfg.max_delay

        # Buffer shape: (max_delay + 1, num_envs, action_dim)
        self._action_buffer = torch.zeros(
            self.max_delay + 1, self.num_envs, self.action_dim, device=self.device
        )

        # Active delays for each env, shape: (num_envs,)
        self._active_delays = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_delays(torch.arange(self.num_envs, device=self.device))

    def reset_delays(self, env_ids: torch.Tensor):
        if self.min_delay == self.max_delay:
            self._active_delays[env_ids] = self.min_delay
        else:
            sampled = torch.randint(
                self.min_delay,
                self.max_delay + 1,
                (len(env_ids),),
                dtype=torch.long,
                device=self.device,
            )
            self._active_delays[env_ids] = sampled

    def process_actions(self, actions: torch.Tensor):
        if self.max_delay > 0:
            self._action_buffer = torch.cat(
                [self._action_buffer[1:], actions.unsqueeze(0)], dim=0
            )
        else:
            self._action_buffer[0] = actions

        indices = self.max_delay - self._active_delays
        actions_delayed = self._action_buffer[
            indices, torch.arange(self.num_envs, device=self.device)
        ]

        filtered = self._weights[0] * actions_delayed + self._weights[1] * self._prev_raw
        self._prev_raw[:] = actions_delayed
        super(JointPositionLowPassAction, self).process_actions(filtered)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]

        self.reset_delays(env_ids)
        self._action_buffer[:, env_ids] = 0.0
        self._prev_raw[env_ids] = 0.0
        super().reset(env_ids)


@dataclass(kw_only=True)
class PostureOffsetPositionActionCfg(JointPositionDelayedLowPassActionCfg):
    """Joint position action whose default offset follows a commanded posture.

    The action target is ``offset + raw * scale``. With the offset pinned to the
    standing stance (the stock behaviour) the folded pose sits seven to nine
    action sigmas away and the policy can never choose to fold. Here the offset
    instead interpolates between the folded pose and the standing stance
    according to the ``posture`` command, so zero action means "hold whatever
    posture was asked for" and only the corrections have to be learned.

    Set ``posture_command_name`` to ``None`` to fall back to the stock fixed
    offset.
    """

    posture_command_name: str | None = "posture"

    def build(self, env: ManagerBasedRlEnv) -> PostureOffsetPositionAction:
        return PostureOffsetPositionAction(self, env)


class PostureOffsetPositionAction(JointPositionDelayedLowPassAction):
    """Interpolates the position offset between folded and standing each step."""

    def __init__(self, cfg: PostureOffsetPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._posture_name = cfg.posture_command_name
        # The stock class already put the nominal stance in self._offset; keep it
        # as the alpha = 1 end of the interpolation.
        self._standing_offset = self._offset.clone()
        self._folded_offset = self._folded_for_targets()

    def _folded_for_targets(self) -> torch.Tensor:
        """Folded leg angles for this term's targets, by joint name."""
        from ..stance import FOLDED_STANCE

        by_kind = {"hip": FOLDED_STANCE[0], "thigh": FOLDED_STANCE[1], "knee": FOLDED_STANCE[2]}
        values = []
        for name in self._target_names:
            angle = 0.0
            for kind, folded in by_kind.items():
                if f"_{kind}_" in name:
                    angle = folded
                    break
            values.append(angle)
        return torch.tensor(values, device=self.device, dtype=self._offset.dtype).expand(
            self.num_envs, -1
        )

    def _sync_offset(self) -> None:
        if self._posture_name is None:
            return
        alpha = self._env.command_manager.get_term(self._posture_name).alpha.unsqueeze(-1)
        self._offset[:] = self._folded_offset + (self._standing_offset - self._folded_offset) * alpha

    def process_actions(self, actions: torch.Tensor):
        self._sync_offset()
        super().process_actions(actions)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._sync_offset()
        super().reset(env_ids)


@dataclass(kw_only=True)
class JointVelocityDelayedLowPassActionCfg(JointVelocityLowPassActionCfg):
    """Joint velocity action with random delay and 1st-order low-pass filter."""

    min_delay: int = 0
    max_delay: int = 2

    def build(self, env: ManagerBasedRlEnv) -> JointVelocityDelayedLowPassAction:
        return JointVelocityDelayedLowPassAction(self, env)


class JointVelocityDelayedLowPassAction(JointVelocityLowPassAction):
    """Applies a random delay and a 1st-order low-pass filter to raw actions."""

    def __init__(self, cfg: JointVelocityDelayedLowPassActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.min_delay = cfg.min_delay
        self.max_delay = cfg.max_delay

        # Buffer shape: (max_delay + 1, num_envs, action_dim)
        self._action_buffer = torch.zeros(
            self.max_delay + 1, self.num_envs, self.action_dim, device=self.device
        )

        # Active delays for each env, shape: (num_envs,)
        self._active_delays = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_delays(torch.arange(self.num_envs, device=self.device))

    def reset_delays(self, env_ids: torch.Tensor):
        if self.min_delay == self.max_delay:
            self._active_delays[env_ids] = self.min_delay
        else:
            sampled = torch.randint(
                self.min_delay,
                self.max_delay + 1,
                (len(env_ids),),
                dtype=torch.long,
                device=self.device,
            )
            self._active_delays[env_ids] = sampled

    def process_actions(self, actions: torch.Tensor):
        if self.max_delay > 0:
            self._action_buffer = torch.cat(
                [self._action_buffer[1:], actions.unsqueeze(0)], dim=0
            )
        else:
            self._action_buffer[0] = actions

        indices = self.max_delay - self._active_delays
        actions_delayed = self._action_buffer[
            indices, torch.arange(self.num_envs, device=self.device)
        ]

        filtered = self._weights[0] * actions_delayed + self._weights[1] * self._prev_raw
        self._prev_raw[:] = actions_delayed
        super(JointVelocityLowPassAction, self).process_actions(filtered)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]

        self.reset_delays(env_ids)
        self._action_buffer[:, env_ids] = 0.0
        self._prev_raw[env_ids] = 0.0
        super().reset(env_ids)

