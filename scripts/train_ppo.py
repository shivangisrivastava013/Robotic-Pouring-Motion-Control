import argparse
import json
import os
import platform
import sys
import time

import gymnasium as gym
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pouring_sim.environment import RoboticPouringEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO policy for RoboticPouringEnv.")
    parser.add_argument("--timesteps", type=int, default=150000, help="Total training timesteps.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save model checkpoint.")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_env = Monitor(RoboticPouringEnv())
    eval_env = Monitor(RoboticPouringEnv())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.save_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    hyperparams = {
        "learning_rate": args.lr,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "seed": args.seed,
    }

    print(f"[+] Training PPO policy for {args.timesteps} timesteps (Seed {args.seed})...")
    start_time = time.time()
    model = PPO("MlpPolicy", train_env, verbose=1, **hyperparams)
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    training_duration = round(time.time() - start_time, 2)

    save_path = os.path.join(args.save_dir, "ppo_pouring.zip")
    best_path = os.path.join(args.save_dir, "best_model.zip")
    if os.path.exists(best_path):
        os.replace(best_path, save_path)
    else:
        model.save(save_path)

    metadata = {
        "algorithm": "PPO",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "hyperparameters": hyperparams,
        "checkpoint_path": save_path,
        "training_duration_seconds": training_duration,
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gym.__version__,
        },
    }

    meta_path = os.path.join(args.save_dir, "ppo_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] PPO Model saved to '{save_path}' with metadata in '{meta_path}'.")


if __name__ == "__main__":
    main()
