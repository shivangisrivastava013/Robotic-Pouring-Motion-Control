import os
import matplotlib.pyplot as plt
from pouring_sim.environment import RoboticPouringEnv


def main():
    print("[+] Initializing Robotic Pouring Motion Control Simulation...")
    os.makedirs("./results", exist_ok=True)

    env = RoboticPouringEnv(target_volume_ml=250.0)
    obs = env.reset()

    history_tilt = []
    history_poured = []
    history_spilled = []

    print("\n[*] Running 30-step Trajectory Simulation Loop:\n")
    print(f"{'Step':<6} | {'Tilt Angle (deg)':<18} | {'Poured (ml)':<14} | {'Spilled (ml)':<14} | {'Accuracy (%)'}")
    print("-" * 75)

    for step in range(1, 31):
        # PID-like smooth acceleration trajectory control
        if env.poured_volume < env.target_volume_ml * 0.8:
            action = 0.6  # Accelerate tilt
        elif env.poured_volume < env.target_volume_ml:
            action = -0.2  # Slow down tilt
        else:
            action = -0.8  # Reverse tilt back up

        obs, reward, done, info = env.step(action)

        history_tilt.append(info["tilt_angle_deg"])
        history_poured.append(info["poured_volume_ml"])
        history_spilled.append(info["spilled_volume_ml"])

        if step % 3 == 0 or step == 1 or done:
            print(f"{step:<6} | {info['tilt_angle_deg']:<18.1f} | {info['poured_volume_ml']:<14.1f} | {info['spilled_volume_ml']:<14.1f} | {info['volume_accuracy_pct']:.1f}%")

        if done:
            break

    print("\n[+] ROBOTIC POURING SIMULATION SUMMARY:")
    print(f"   - Target Volume Goal:   250.0 ml")
    print(f"   - Liquid Poured:        {info['poured_volume_ml']} ml")
    print(f"   - Fluid Spilled:        {info['spilled_volume_ml']} ml")
    print(f"   - Pouring Precision:    {info['volume_accuracy_pct']}%")

    # Plot simulation results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(history_tilt, color="#00f2fe", linewidth=2, label="Container Tilt Angle (deg)")
    ax1.axhline(y=45.0, color="gray", linestyle="--", alpha=0.7, label="Pouring Threshold (45°)")
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("Robotic Joint Angular Trajectory")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history_poured, color="#4facfe", linewidth=2, label="Poured Liquid Volume (ml)")
    ax2.axhline(y=250.0, color="green", linestyle="--", label="Target Goal (250 ml)")
    ax2.plot(history_spilled, color="#ff0844", linewidth=2, label="Spilled Liquid (ml)")
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Volume (ml)")
    ax2.set_title("Liquid Volume Transfer & Spill Control")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = "./results/robotic_pouring_trajectory.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\n[+] Saved trajectory plot to: {output_path}")


if __name__ == '__main__':
    main()
