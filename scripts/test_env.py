"""Quick test: create DoomEnv, step through a few tics with random actions."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import DoomEnv


def main() -> None:
    env = DoomEnv(visible=False, frame_skip=4)
    print(f"Action space size: {env.num_actions}")

    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")

    rng = np.random.default_rng(seed=0)
    total_reward = 0.0

    for step_i in range(20):
        action = int(rng.integers(env.num_actions))
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(
                f"  Step {step_i + 1}: episode finished | reward={reward:.1f} | info={info}"
            )
            break
        else:
            print(
                f"  Step {step_i + 1}: obs={obs.shape} | reward={reward:.1f} | info={info}"
            )

    print(f"\nTotal reward over {step_i + 1} steps: {total_reward:.2f}")
    env.close()
    print("✓ DoomEnv test passed")


if __name__ == "__main__":
    main()
