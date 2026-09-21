import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from scripts.compare_controllers import main as run_benchmark

if __name__ == "__main__":
    print("[+] Initializing Robotic Pouring Motion Control Benchmark Engine...")
    run_benchmark()
