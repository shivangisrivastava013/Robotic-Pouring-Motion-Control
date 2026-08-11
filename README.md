# Robotic Pouring Motion Control via Reinforcement Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-Physics_Engine-FF6F00?style=for-the-badge)](https://pybullet.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Robotic joint trajectory planning and liquid container pouring simulation utilizing **Deep Reinforcement Learning (PPO / SAC)** and physics-based motion control for precise fluid volume control and spill avoidance.

---

## 🌟 Features
- 🧪 **Physics Fluid Simulation:** Simulates container angular tilt dynamics, fluid velocity, target liquid volume error, and spill penalty loss functions.
- 🤖 **Deep RL Control Policy:** Proximal Policy Optimization (PPO) & Soft Actor-Critic (SAC) neural policies for continuous torque and joint trajectory regulation.
- 📈 **Trajectory Analytics:** Automated plotting of joint angular tilt vs poured fluid volume.

---

## 📁 Repository Structure
```text
Robotic-Pouring-Motion-Control/
├── pouring_sim/            # Simulation & RL Environment Package
│   ├── __init__.py
│   └── environment.py      # Physics Pouring Environment
├── demo.py                 # 1-Command Trajectory Simulation Loop
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start
```bash
git clone https://github.com/shivangisrivastava013/Robotic-Pouring-Motion-Control.git
cd Robotic-Pouring-Motion-Control

pip install -r requirements.txt
python demo.py
```

---

## 👤 Author
**Shivangi Srivastava**  
MS in Artificial Intelligence @ NJIT  
[LinkedIn Profile](https://www.linkedin.com/in/shivangisrivastava013/) | [Portfolio](https://shivangisrivastava013.github.io/shivangi-portfolio/)
