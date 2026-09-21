import argparse
import os
import sys

from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pouring_sim.environment import RoboticPouringEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO policy for RoboticPouringEnv.")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save model checkpoint.")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env = RoboticPouringEnv()

    print(f"[+] Training PPO policy for {args.timesteps} timesteps (Seed {args.seed})...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        seed=args.seed,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps)

    save_path = os.path.join(args.save_dir, "ppo_pouring.zip")
    model.save(save_path)
    print(f"[+] PPO Model saved to '{save_path}'.")


if __name__ == "__main__":
    main()
