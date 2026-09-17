from mjlab.tasks.registry import register_mjlab_task
from .config.env_cfgs import flat_env_cfg, rough_env_cfg, recovery_env_cfg
from .config.rl_cfg import flat_ppo_runner_cfg, rough_ppo_runner_cfg, recovery_ppo_runner_cfg

register_mjlab_task(task_id="Wheelleg-Flat-v0", env_cfg=flat_env_cfg(), play_env_cfg=flat_env_cfg(play=True), rl_cfg=flat_ppo_runner_cfg())
register_mjlab_task(task_id="Wheelleg-Rough-v0", env_cfg=rough_env_cfg(), play_env_cfg=rough_env_cfg(play=True), rl_cfg=rough_ppo_runner_cfg())
register_mjlab_task(task_id="Wheelleg-Recovery-v0", env_cfg=recovery_env_cfg(), play_env_cfg=recovery_env_cfg(play=True), rl_cfg=recovery_ppo_runner_cfg())
