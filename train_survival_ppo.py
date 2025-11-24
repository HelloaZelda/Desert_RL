import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from desert_env_survival import DesertGoldSurvivalEnv


# -------------------------------------------
# Create training environment
# -------------------------------------------
def make_env():
    env = DesertGoldSurvivalEnv(
        edges_path="map_edges.json",
        nodes_path="map_nodes.json",
        max_days=50
    )
    return Monitor(env)


env = DummyVecEnv([make_env])


# -------------------------------------------
# PPO Configuration (stable for this env)
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

# directories
models_dir = "models_survival"
logs_dir = "rollout_logs_survival"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)


# -------------------------------------------
# Utility: record trajectory for analysis
# -------------------------------------------
def record_trajectory(obs, action, reward, done):
    with open(f"{logs_dir}/trajectory.txt", "a") as f:
        f.write(f"{obs.tolist()} | a={action} | r={reward} | done={done}\n")


# -------------------------------------------
# Training Loop
# -------------------------------------------
TIMESTEPS = 10000
TOTAL_TIMESTEPS = 500_000

current_steps = 0

while current_steps < TOTAL_TIMESTEPS:

    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False
    )
    current_steps += TIMESTEPS

    # Save model
    model_path = f"{models_dir}/ppo_survival_{current_steps}"
    model.save(model_path)
    print(f"Saved model: {model_path}")

    # ---------------------------------------
    # Evaluate once after each training cycle
    # ---------------------------------------
    eval_env = DesertGoldSurvivalEnv(
        edges_path="map_edges.json",
        nodes_path="map_nodes.json"
    )
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward

        record_trajectory(obs, action, reward, done)
        obs = next_obs

    print(f"[Eval] Steps={current_steps} TotalReward={total_reward}")


print("Training finished!")
