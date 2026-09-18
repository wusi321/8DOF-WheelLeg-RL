from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

def _runner(name: str, iterations: int) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True, distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "log"}),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.002, num_learning_epochs=5, num_mini_batches=4, learning_rate=5e-4, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
        experiment_name=name, save_interval=100, num_steps_per_env=24, max_iterations=iterations,
    )

def flat_ppo_runner_cfg(): return _runner("wheelleg_flat", 6000)
def rough_ppo_runner_cfg(): return _runner("wheelleg_rough", 12000)
def recovery_ppo_runner_cfg(): return _runner("wheelleg_recovery", 8000)
