import argparse
import os
import sys

from stable_baselines3 import PPO, SAC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pouring_sim.controllers import (
    ConstantActionController,
    PIDController,
    RuleBasedController,
    SB3PolicyWrapper,
)
from pouring_sim.environment import RoboticPouringEnv
from pouring_sim.evaluation import evaluate_controllers


def main():
    parser = argparse.ArgumentParser(description="Run full controller benchmark (Constant, Rule-Based, PID, PPO, SAC).")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes per controller.")
    parser.add_argument("--timesteps", type=int, default=30000, help="RL training timesteps per model.")
    parser.add_argument("--mock-train", action="store_true", help="Run with short timesteps for smoke testing.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory for output files.")

    args = parser.parse_args()

    env = RoboticPouringEnv()

    # Train or initialize RL models
    train_steps = 2000 if args.mock_train else args.timesteps
    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ppo_path = os.path.join(ckpt_dir, "ppo_pouring.zip")
    sac_path = os.path.join(ckpt_dir, "sac_pouring.zip")

    if os.path.exists(ppo_path) and not args.mock_train:
        print(f"[+] Loading PPO checkpoint from '{ppo_path}'...")
        ppo_model = PPO.load(ppo_path, env=env)
    else:
        print(f"[+] Training PPO policy for {train_steps} timesteps...")
        ppo_model = PPO("MlpPolicy", env, learning_rate=3e-4, seed=42, verbose=0)
        ppo_model.learn(total_timesteps=train_steps)
        ppo_model.save(ppo_path)

    if os.path.exists(sac_path) and not args.mock_train:
        print(f"[+] Loading SAC checkpoint from '{sac_path}'...")
        sac_model = SAC.load(sac_path, env=env)
    else:
        print(f"[+] Training SAC policy for {train_steps} timesteps...")
        sac_model = SAC("MlpPolicy", env, learning_rate=3e-4, seed=42, verbose=0)
        sac_model.learn(total_timesteps=train_steps)
        sac_model.save(sac_path)

    controllers = [
        ConstantActionController(0.3),
        RuleBasedController(target_volume=200.0),
        PIDController(kp=2.5, ki=0.05, kd=0.8),
        SB3PolicyWrapper(ppo_model, name="PPO (RL)"),
        SB3PolicyWrapper(sac_model, name="SAC (RL)"),
    ]

    evaluate_controllers(
        env,
        controllers,
        num_episodes=args.episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
