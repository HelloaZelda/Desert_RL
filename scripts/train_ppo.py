import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from envs.desert_env_full import DesertGoldEnv


DATA_DIR = ROOT / "data"


# -------------------------------------------
# Create training environment
# -------------------------------------------
def make_env():
    env = DesertGoldEnv(
        edges_path=str(DATA_DIR / "map_edges.json"),
        nodes_path=str(DATA_DIR / "map_nodes.json"),
        num_teams=1,
        max_days=40
    )
    env = Monitor(env)
    return env


# Vec wrapper
env = DummyVecEnv([make_env])


# -------------------------------------------
# PPO Hyperparameters
# -------------------------------------------
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

# -------------------------------------------
# Training loop with autosave
# -------------------------------------------
TIMESTEPS = 10000
TOTAL = 500_000   # train 5e5 steps (change as needed)

models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

log_dir = "rollout_logs"
os.makedirs(log_dir, exist_ok=True)

# Record trajectories for later visualization
def record_trajectory(obs, action, reward, done):
    fname = os.path.join(log_dir, "trajectory.txt")
    with open(fname, "a") as f:
        f.write(f"{obs.tolist()} | a={action} | r={reward} | done={done}\n")


# -------------------------------------------
# Main training
# -------------------------------------------
step = 0
while step < TOTAL:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)

    step += TIMESTEPS

    # Save model
    model.save(f"{models_dir}/ppo_{step}")

    # Evaluate environment once and record a trajectory
    eval_env = DesertGoldEnv(
        edges_path=str(DATA_DIR / "map_edges.json"),
        nodes_path=str(DATA_DIR / "map_nodes.json")
    )
    obs, _ = eval_env.reset()
    done = False
    total_r = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

        total_r += reward
        record_trajectory(obs, action, reward, done)

        obs = next_obs

    print(f"[Eval] step={step}, reward={total_r}")


print("Training complete!")
