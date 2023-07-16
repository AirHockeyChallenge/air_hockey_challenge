import os

if __name__ == "__main__":
    from experiment_launcher.launcher import Launcher

    launcher = Launcher(exp_name="AirHockey-SAC-Defend-Acc", n_seeds=3, exp_file="air_hockey_exp",
                        memory_per_core=3000, conda_env="challenge", n_exps_in_parallel=3,
                        hours=32, minutes=00, use_timestamp=True, base_dir='logs')

    defaults = {
        "actor_lr": 1e-4,
        "critic_lr": 3e-4,
        "n_features": "128 128 128",
        "batch_size": 64,
        "initial_replay_size": 50000,
        "max_replay_size": 200000,
        "tau": 1e-3,
        "warmup_transitions": 10000,
        "lr_alpha": 1e-5,
        "target_entropy": -7,
        "use_cuda": False,

        "env": "7dof-hit",
        "quiet": True,
        "alg": "sac-atacom",
        "checkpoint": "None",

        "n_steps": 50000,
        "n_epochs": 100,

        "interpolation_order": -1,

        "double_integration": True,
    }

    use_wandb = True
    if use_wandb:
        import wandb

        wandb_key = os.getenv("WANDB_KEY_PUZE")
        wandb.login(key=wandb_key, relogin=True)

    lr_alpha = [5e-6]
    for lr in lr_alpha:
        defaults["lr_alpha"] = lr
        launcher.add_experiment(**defaults)

    launcher.run(True)
