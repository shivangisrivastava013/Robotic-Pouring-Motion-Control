import os

import imageio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_controller_comparisons(df_summary: pd.DataFrame, output_dir: str = "./results"):
    """
    Plots benchmark comparison charts for Success Rate, Volume Error, and Spill Volume.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Success Rate Comparison
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df_summary, x="controller", y="success_rate", hue="controller", palette="viridis", legend=False
    )
    plt.title("Controller Success Rate (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Controller", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(0, 105)

    for p in ax.patches:
        height = p.get_height()
        if height >= 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate.png"), dpi=300)
    plt.close()

    # 2. Volume Error Comparison
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df_summary, x="controller", y="mean_absolute_error_ml", hue="controller", palette="rocket", legend=False
    )
    plt.title("Mean Absolute Volume Error (ml)", fontsize=14, fontweight="bold")
    plt.xlabel("Controller", fontsize=12)
    plt.ylabel("Absolute Volume Error (ml)", fontsize=12)

    for p in ax.patches:
        height = p.get_height()
        if height >= 0:
            ax.annotate(
                f"{height:.2f} ml",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "volume_error.png"), dpi=300)
    plt.close()

    # 3. Spill Volume Comparison
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df_summary, x="controller", y="mean_spill_ml", hue="controller", palette="mako", legend=False)
    plt.title("Mean Spill Volume (ml)", fontsize=14, fontweight="bold")
    plt.xlabel("Controller", fontsize=12)
    plt.ylabel("Cumulative Spill (ml)", fontsize=12)

    for p in ax.patches:
        height = p.get_height()
        if height >= 0:
            ax.annotate(
                f"{height:.2f} ml",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spill_comparison.png"), dpi=300)
    plt.close()


def plot_trajectory_examples(trajectories: dict, output_dir: str = "./results"):
    """
    Plots state trajectories over step time for each controller.
    """
    os.makedirs(output_dir, exist_ok=True)
    _, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for name, traj in trajectories.items():
        steps = range(len(traj["tilt_deg"]))
        axes[0].plot(steps, traj["poured"], label=name, linewidth=2)
        axes[1].plot(steps, traj["tilt_deg"], label=name, linewidth=2)
        axes[2].plot(steps, traj["spill"], label=name, linewidth=2)

    axes[0].set_ylabel("Poured Volume (ml)", fontsize=11)
    axes[0].axhline(y=200.0, color="red", linestyle="--", label="Target Volume (200 ml)")
    axes[0].set_title("Container Poured Volume Profile", fontsize=12, fontweight="bold")
    axes[0].legend()

    axes[1].set_ylabel("Tilt Angle (deg)", fontsize=11)
    axes[1].axhline(y=45.0, color="gray", linestyle=":", label="Flow Threshold (45°)")
    axes[1].set_title("Robotic Pouring Tilt Profile", fontsize=12, fontweight="bold")
    axes[1].legend()

    axes[2].set_ylabel("Cumulative Spill (ml)", fontsize=11)
    axes[2].set_xlabel("Simulation Step", fontsize=11)
    axes[2].set_title("Spill Accumulation Profile", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_examples.png"), dpi=300)
    plt.close()


def create_pouring_demo_gif(env, controller, output_path: str = "./results/pouring_demo.gif"):
    """
    Renders an animated demonstration GIF of a pouring episode.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []

    obs, info = env.reset(seed=42)
    controller.reset()

    for _ in range(100):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = controller.select_action(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    if frames:
        imageio.mimsave(output_path, frames, fps=10, loop=0)
