import json
import os
import time

import numpy as np
import pandas as pd

from pouring_sim.visualization import create_pouring_demo_gif, plot_controller_comparisons, plot_trajectory_examples


def evaluate_controllers(
    env,
    controllers: list,
    num_episodes: int = 100,
    target_volumes: list | None = None,
    output_dir: str = "./results",
):
    """
    Evaluates controllers across multi-trial randomized episodes.
    Calculates Success Rate, MAE, Median Error, Spill Volume, Overshoot Rate, Duration, and Latency.
    """
    target_vols = target_volumes or [100.0, 150.0, 200.0, 250.0, 300.0]
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    full_json_results = {}
    sample_trajectories = {}

    print(f"[+] Starting Multi-Trial Controller Benchmark ({num_episodes} Episodes/Controller)...")

    for ctrl in controllers:
        name = ctrl.name
        print(f"\n[>] Evaluating Controller: {name}")

        successes = []
        volume_errors = []
        spills = []
        overshoots = []
        durations = []
        rewards = []
        latencies = []

        demo_traj = {"poured": [], "tilt_deg": [], "spill": []}

        for ep in range(num_episodes):
            target_v = target_vols[ep % len(target_vols)]
            obs, info = env.reset(seed=42 + ep, options={"target_volume": target_v})
            ctrl.reset()

            ep_reward = 0.0
            steps = 0

            while True:
                start_t = time.perf_counter()
                action = ctrl.select_action(obs, info)
                latencies.append((time.perf_counter() - start_t) * 1000.0)

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1

                if ep == 0:
                    demo_traj["poured"].append(info["poured_volume_ml"])
                    demo_traj["tilt_deg"].append(info["tilt_angle_deg"])
                    demo_traj["spill"].append(info["cumulative_spill_ml"])

                if terminated or truncated:
                    break

            vol_err = abs(info["poured_volume_ml"] - info["target_volume_ml"])
            is_overshoot = info["poured_volume_ml"] > (info["target_volume_ml"] + 5.0)

            successes.append(info.get("is_success", False))
            volume_errors.append(vol_err)
            spills.append(info["cumulative_spill_ml"])
            overshoots.append(is_overshoot)
            durations.append(steps)
            rewards.append(ep_reward)

        sample_trajectories[name] = demo_traj

        stats = {
            "controller": name,
            "success_rate": round(float(np.mean(successes) * 100.0), 2),
            "mean_absolute_error_ml": round(float(np.mean(volume_errors)), 2),
            "median_error_ml": round(float(np.median(volume_errors)), 2),
            "mean_spill_ml": round(float(np.mean(spills)), 2),
            "overshoot_rate": round(float(np.mean(overshoots) * 100.0), 2),
            "mean_duration_steps": round(float(np.mean(durations)), 1),
            "mean_reward": round(float(np.mean(rewards)), 2),
            "inference_latency_ms": round(float(np.mean(latencies)), 3),
        }

        summary_rows.append(stats)
        full_json_results[name] = stats

        print(
            f"    Result: Success Rate = {stats['success_rate']}% | "
            f"MAE = {stats['mean_absolute_error_ml']} ml | Spill = {stats['mean_spill_ml']} ml"
        )

    # Export CSV & JSON
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "controller_comparison.csv")
    df_summary.to_csv(csv_path, index=False)

    export_json_data = {
        "metadata": {
            "num_episodes": num_episodes,
            "target_volumes_ml": target_vols,
            "environment": "RoboticPouringEnv (Gymnasium-compliant synthetic numerical pouring simulation)",
            "evaluation_seed_base": 42,
        },
        "results": full_json_results,
    }

    json_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(export_json_data, f, indent=2)

    # Plot Comparison Artifacts
    plot_controller_comparisons(df_summary, output_dir=output_dir)
    plot_trajectory_examples(sample_trajectories, output_dir=output_dir)

    # Generate Animated GIF for best controller (PID or RL)
    best_ctrl = controllers[-1]
    create_pouring_demo_gif(env, best_ctrl, output_path=os.path.join(output_dir, "pouring_demo.gif"))

    print(f"\n[+] Controller Benchmark Complete! Artifacts saved to '{output_dir}'.")
    return df_summary
