# Desert RL

This repository contains reinforcement learning environments and training scripts for the Desert Gold survival game. The main environment `DesertGoldSurvivalEnv` implements the A.2 rule set described in the bundled rules document, including phased progression (outbound, mining, return), weather-driven survival pressure, inventory weight tracking, and shaped rewards that encourage reaching and mining the gold site before returning safely to camp.

## Repository layout
- `desert_env_survival.py` — single-agent Gymnasium environment following the A.2 survival rules with dict observations and 25-action space.
- `desert_env_full.py` — earlier/full-version environment with additional features beyond A.2.
- `train_survival_ppo.py` — example PPO training script using the survival environment and a `MultiInputPolicy` compatible with dict observations.
- `train_ppo.py` — baseline PPO training driver for the full environment.
- `map_edges.json` and `map_nodes.json` — map topology and node metadata used by the environments.
- `map_analysis.py` and `map_visualize.py` — utilities for inspecting and visualizing the map structure.

## Quick start
1. Install dependencies (Gymnasium, Stable Baselines3, and supporting libraries). A minimal setup:
   ```bash
   pip install gymnasium==0.29.1 stable-baselines3==2.3.2 torch
   ```
2. Verify the code compiles:
   ```bash
   python -m compileall desert_env_survival.py train_survival_ppo.py
   ```
3. Train an agent on the survival environment:
   ```bash
   python train_survival_ppo.py
   ```
   The script loads the map files, instantiates `DesertGoldSurvivalEnv`, and runs PPO training with default hyperparameters.

## Gameplay summary (A.2 rules)
- **Phases:** Outbound (camp → mine), mining (at mine), and return (mine → camp), with rewards for reaching the mine, mining, beginning return, and arriving back at camp.
- **Actions:** 25 discrete actions; movement uses neighbor indexing, shop/rest actions are available at camp or villages, and mining is only allowed at the mine.
- **Resources & weight:** Food, water, gold, tent uses, and compass stock all contribute to total carried weight (limit 1000 lbs). Overweight moves are invalid.
- **Weather:** Daily weather affects consumption and can lock the agent during sandstorms unless a tent or compass is used; storms combined with heat have the strongest penalties.
- **Survival:** Running out of food or water outside an oasis ends the episode; oases allow drinking for the day but water cannot be carried away.

Refer to `desert_env_survival.py` for the exact implementation of these mechanics and to `train_survival_ppo.py` for an end-to-end training example.
