import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random


class DesertGoldSurvivalEnv(gym.Env):
    """
    单人强化学习环境：学习挖金、生存、补给、沙暴处理、回营地。
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        edges_path="map_edges.json",
        nodes_path="map_nodes.json",
        max_days=50,
        seed=42
    ):
        super().__init__()

        random.seed(seed)
        np.random.seed(seed)

        # Load maps
        self.edges = json.load(open(edges_path))
        self.nodes = json.load(open(nodes_path))

        self.max_days = max_days
        self.max_weight = 1000

        # Weather state
        self.storm_days_left = 0
        self.day_count = 1
        self.is_day = True

        # Observation space (17维向量)
        # [day, is_day, node_id, food, water, gold, money,
        #  hp, weight, compass, tent_uses,
        #  storm_days_left, is_heat,
        #  near_mine, near_tomb, node_type, neighbor_cnt]
        self.observation_space = spaces.Box(
            low=0, high=2000, shape=(17,), dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(18)
        # 0-7: move
        # 8: mine
        # 9: rest
        # 10: buy_food
        # 11: buy_water
        # 12: buy_tent
        # 13: buy_compass
        # 14: dummy (决斗占位)
        # 15: dummy
        # 16: dummy
        # 17: return_camp

        self.reset_all_states()

    # ---------------------------
    # Helpers
    # ---------------------------
    def node_type(self, node):
        val = self.nodes.get(node, {"type": "S"})
        return val["type"]

    def compute_weight(self):
        w = (
            self.food * 10
            + self.water * 25
            + self.gold * 2
            + self.compass * 10
        )
        if self.tent_uses > 0:
            w += self.tent_weight
        return w

    def check_weight(self):
        while self.compute_weight() > self.max_weight:
            if self.gold > 0:
                self.gold -= 1
            else:
                break

    # ---------------------------
    # Reset
    # ---------------------------
    def reset_all_states(self):
        self.pos = "CAMP"
        self.food = 5
        self.water = 5
        self.gold = 0
        self.money = 100
        self.hp = 10

        self.compass = 0
        self.tent_uses = 0
        self.tent_weight = 0

        self.storm_days_left = 0
        self.day_count = 1
        self.is_day = True

        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_all_states()
        return self._obs(), {}

    # ---------------------------
    # Sandstorm / heat
    # ---------------------------
    def step_weather(self):
        # 10% chance daily
        if random.random() < 0.10:
            self.storm_days_left = 3
        else:
            self.storm_days_left = max(0, self.storm_days_left - 1)

    def is_sandstorm(self, node):
        return (self.node_type(node) == "S") and (self.storm_days_left > 0)

    # ---------------------------
    # Step
    # ---------------------------
    def step(self, action):

        if self.done:
            return self._obs(), 0, True, False, {}

        node = self.pos
        reward = 0

        # If sandstorm and no compass/tent -> frozen
        if self.is_sandstorm(node) and self.compass == 0 and self.tent_uses == 0:
            if self.storm_days_left > 0:
                reward -= 2
                self._advance_time()
                return self._obs(), reward, False, False, {}

        # ------------- Actions -------------
        if 0 <= action <= 7:
            reward += self._act_move(action)

        elif action == 8:
            reward += self._act_mine()

        elif action == 9:
            reward += self._act_rest()

        elif action == 10:
            reward += self._act_buy_food()

        elif action == 11:
            reward += self._act_buy_water()

        elif action == 12:
            reward += self._act_buy_tent()

        elif action == 13:
            reward += self._act_buy_compass()

        elif action == 17:
            reward += self._act_return_camp()

        else:
            reward -= 1

        # ------------- Consumption -------------
        reward += self._apply_consumption()

        # ------------- Weight -------------
        self.check_weight()

        # ------------- End-of-day -------------
        self._advance_time()

        # death check
        if self.hp <= 0:
            self.done = True
            return self._obs(), -50, True, False, {}

        if self.done:
            return self._obs(), reward, True, False, {}

        return self._obs(), reward, False, False, {}

    # ---------------------------
    # Actions
    # ---------------------------
    def _act_move(self, action):
        neighbors = self.edges[self.pos]
        if action >= len(neighbors):
            return -1
        self.pos = neighbors[action]
        return 0.5

    def _act_mine(self):
        if self.node_type(self.pos) != "MINE":
            return -1
        self.gold += 25
        return 1.0

    def _act_rest(self):
        if self.tent_uses > 0:
            self.tent_uses -= 1
            self.tent_weight = max(0, self.tent_weight - 20)
        return 0.2

    def _act_buy_food(self):
        if self.node_type(self.pos) != "C":
            return -1
        if self.money < 20:
            return -1
        self.money -= 20
        self.food += 1
        return 0.2

    def _act_buy_water(self):
        if self.node_type(self.pos) != "C":
            return -1
        if self.money < 20:
            return -1
        self.money -= 20
        self.water += 1
        return 0.2

    def _act_buy_tent(self):
        if self.node_type(self.pos) != "C":
            return -1
        if self.money < 400:
            return -1
        self.money -= 400
        self.tent_uses = 3
        self.tent_weight = 70
        return 0.3

    def _act_buy_compass(self):
        if self.node_type(self.pos) != "C":
            return -1
        if self.money < 100:
            return -1
        self.money -= 100
        self.compass += 1
        return 0.2

    def _act_return_camp(self):
        if self.pos != "CAMP":
            return -1
        self.money += self.gold * 10
        self.gold = 0
        self.done = True
        return 20

    # ---------------------------
    # Consumption
    # ---------------------------
    def _apply_consumption(self):
        node_t = self.node_type(self.pos)
        reward = 0

        # Oasis
        if node_t == "L":
            w_cost = 0
            f_cost = 1
        else:
            w_cost = 1
            f_cost = 1

            # heatwave double water
            if node_t in ["S", "L"]:
                w_cost *= 2

            if self.is_sandstorm(self.pos) and self.compass == 0 and self.tent_uses == 0:
                w_cost += 1

        # tent reduces consumption at night
        if not self.is_day and self.tent_uses > 0:
            f_cost = 0
            w_cost = 0

        self.food = max(0, self.food - f_cost)
        self.water = max(0, self.water - w_cost)

        if self.food == 0 or self.water == 0:
            self.hp -= 1
            reward -= 5

        return reward

    # ---------------------------
    # Observation
    # ---------------------------
    def _obs(self):
        node = self.pos
        neighbors = self.edges[node]

        near_mine = 1 if any(self.node_type(x) == "MINE" for x in neighbors) else 0
        near_tomb = 1 if any(self.node_type(x) == "TOMB" for x in neighbors) else 0

        node_t = self.node_type(node)
        type_id = {"S": 0, "C": 1, "L": 2, "CAMP": 3, "MINE": 4, "TOMB": 5}[node_t]

        return np.array([
            self.day_count,
            1 if self.is_day else 0,
            hash(node) % 10000,
            self.food,
            self.water,
            self.gold,
            self.money,
            self.hp,
            self.compute_weight(),
            self.compass,
            self.tent_uses,
            self.storm_days_left,
            1 if node_t in ["S", "L"] else 0,  # heat
            near_mine,
            near_tomb,
            type_id,
            len(neighbors)
        ], dtype=np.float32)

    # ---------------------------
    # Time Advance
    # ---------------------------
    def _advance_time(self):
        if self.is_day:
            self.is_day = False
        else:
            self.is_day = True
            self.day_count += 1
            self.step_weather()
