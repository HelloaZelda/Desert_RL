import json
import random
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DesertGoldSurvivalEnv(gym.Env):
    """Single-agent survival environment implementing the A.2 rule set.

    The environment follows the specification provided in the consolidated
    "Desert Gold" rules (map structure, weather, items, survival, mining, and
    return logic).  Observation space is a Dict capturing the important game
    state values so that agents can reason about phase, weather, inventory, and
    remaining lock time.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Weather codes
    NORMAL = 0
    HEAT = 1
    SANDSTORM = 2
    STORM_HEAT = 3

    def __init__(
        self,
        edges_path: str = "map_edges.json",
        nodes_path: str = "map_nodes.json",
        max_days: int = 50,
        seed: int = 42,
    ) -> None:
        super().__init__()

        random.seed(seed)
        np.random.seed(seed)

        # Load maps
        self.edges: Dict[str, list] = json.load(open(edges_path))
        self.nodes: Dict[str, Dict[str, str]] = json.load(open(nodes_path))
        self.node_to_idx = {name: idx for idx, name in enumerate(self.nodes.keys())}

        # Limits
        self.max_days = max_days
        self.max_weight = 1000

        # Economy / weights
        self.costs_camp = {"food": 10, "water": 25, "tent": 400, "compass": 100}
        self.costs_village = {"food": 20, "water": 50}
        self.weights = {
            "food": 10,
            "water": 50,
            "tent": 70,
            "compass": 10,
        }

        # Action space: 25 discrete actions (0-7 move, 8-14 shop/rest/mine, rest invalid)
        self.action_space = spaces.Discrete(25)

        # Observation space follows the document fields
        self.observation_space = spaces.Dict(
            {
                "day": spaces.Box(0, self.max_days + 10, shape=(1,), dtype=np.int32),
                "node": spaces.Box(0, len(self.nodes), shape=(1,), dtype=np.int32),
                "food": spaces.Box(0, 200, shape=(1,), dtype=np.int32),
                "water": spaces.Box(0, 200, shape=(1,), dtype=np.int32),
                "weight": spaces.Box(0, 2000, shape=(1,), dtype=np.int32),
                "money": spaces.Box(0, 5000, shape=(1,), dtype=np.int32),
                "gold": spaces.Box(0, 2000, shape=(1,), dtype=np.int32),
                "tent": spaces.Box(0, 3, shape=(1,), dtype=np.int32),
                "compass": spaces.Box(0, 5, shape=(1,), dtype=np.int32),
                "locked": spaces.Box(0, 3, shape=(1,), dtype=np.int32),
                "weather": spaces.Box(0, 3, shape=(1,), dtype=np.int32),
                "phase": spaces.Box(0, 2, shape=(1,), dtype=np.int32),
            }
        )

        self.reset_all_states()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def node_type(self, node: str) -> str:
        return self.nodes.get(node, {}).get("type", "S")

    def compute_weight(self) -> int:
        weight = (
            sum(self.food_units)
            + sum(self.water_units)
            + self.gold
            + self.compass * self.weights["compass"]
        )
        if self.tent_uses > 0:
            weight += self.tent_weight
        return weight

    def food_count(self) -> int:
        return len(self.food_units)

    def water_count(self) -> int:
        return len(self.water_units)

    def reset_all_states(self) -> None:
        self.pos = "CAMP"
        self.food_units = [self.weights["food"] for _ in range(5)]
        self.water_units = [self.weights["water"] for _ in range(5)]
        self.gold = 0  # weight in lbs
        self.money = 100
        self.tent_uses = 0
        self.tent_weight = 0
        self.compass = 0

        self.phase = 0  # 0 outbound, 1 mining, 2 return
        self.day_count = 1
        self.weather = self._generate_weather()
        self.locked_days = 0
        self.lock_weather = None

        self.visited = set([self.pos])
        self.done = False

    # ------------------------------------------------------------------
    # Weather handling
    # ------------------------------------------------------------------
    def _generate_weather(self) -> int:
        """Sample weather for the day.

        Probabilities favor normal conditions while still surfacing storms.
        """
        roll = random.random()
        if roll < 0.6:
            return self.NORMAL
        if roll < 0.8:
            return self.HEAT
        if roll < 0.95:
            return self.SANDSTORM
        return self.STORM_HEAT

    def _current_weather(self) -> int:
        return self.lock_weather if self.locked_days > 0 else self.weather

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_all_states()
        return self._obs(), {}

    def step(self, action: int):
        if self.done:
            return self._obs(), 0.0, True, False, {}

        reward = 0.0
        info = {}
        terminated = False
        truncated = False

        current_weather = self._current_weather()
        node_type = self.node_type(self.pos)

        # Attempted movement while locked
        if self.locked_days > 0 and 0 <= action <= 7:
            reward -= 5
        else:
            reward += self._dispatch_action(action)

        # Survival consumption and locking logic
        tool_used, lock_added = self._handle_weather(current_weather, node_type)
        reward += -5 if lock_added else 0

        reward += self._apply_consumption(current_weather, node_type, tool_used)

        # Death rule
        if self._is_dead(node_type):
            reward -= 300
            terminated = True

        # Success
        if self.pos == "CAMP" and self.phase == 2 and not terminated:
            reward += 300
            self.money += (self.gold // 50) * 100
            self.gold = 0
            terminated = True

        # Day advance
        if not terminated:
            self.day_count += 1
            if self.day_count > self.max_days:
                truncated = True
            if self.locked_days > 0:
                self.locked_days = max(0, self.locked_days - 1)
                if self.locked_days == 0:
                    self.lock_weather = None
                    self.weather = self._generate_weather()
            else:
                self.weather = self._generate_weather()

        self.done = terminated or truncated
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------
    def _dispatch_action(self, action: int) -> float:
        if 0 <= action <= 7:
            return self._act_move(action)
        if action == 8:
            return self._act_buy_food()
        if action == 9:
            return self._act_buy_water()
        if action == 10:
            return self._act_buy_tent()
        if action == 11:
            return self._act_buy_compass()
        if action == 12:
            return self._act_buy_info()
        if action == 13:
            return self._act_rest()
        if action == 14:
            return self._act_mine()
        # reserved / invalid
        return -10.0

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _act_move(self, action: int) -> float:
        neighbors = self.edges[self.pos]
        if action >= len(neighbors):
            return -10.0

        if self.compute_weight() > self.max_weight:
            return -10.0

        prev_node = self.pos
        self.pos = neighbors[action]
        reward = 0.0

        # visiting new node reward
        if self.pos not in self.visited:
            self.visited.add(self.pos)
            reward += 1

        # phase transitions
        if self.pos == "MINE" and self.phase == 0:
            self.phase = 1
            reward += 50
        if prev_node == "MINE" and self.pos != "MINE" and self.phase == 1:
            self.phase = 2
            reward += 20

        return reward

    def _act_buy_food(self) -> float:
        node_type = self.node_type(self.pos)
        if node_type == "CAMP":
            cost = self.costs_camp["food"]
            weight = self.weights["food"]
        elif node_type == "C":
            cost = self.costs_village["food"]
            weight = 20
        else:
            return -10.0

        if self.money < cost:
            return -10.0
        self.money -= cost
        self.food_units.append(weight)
        return 0.2

    def _act_buy_water(self) -> float:
        node_type = self.node_type(self.pos)
        if node_type == "CAMP":
            cost = self.costs_camp["water"]
            weight = self.weights["water"]
        elif node_type == "C":
            cost = self.costs_village["water"]
            weight = 100
        else:
            return -10.0

        if self.money < cost:
            return -10.0
        self.money -= cost
        self.water_units.append(weight)
        return 0.2

    def _act_buy_tent(self) -> float:
        if self.node_type(self.pos) != "CAMP":
            return -10.0
        cost = self.costs_camp["tent"]
        if self.money < cost:
            return -10.0
        self.money -= cost
        self.tent_uses = 3
        self.tent_weight = self.weights["tent"]
        return 0.3

    def _act_buy_compass(self) -> float:
        if self.node_type(self.pos) != "CAMP":
            return -10.0
        cost = self.costs_camp["compass"]
        if self.money < cost:
            return -10.0
        self.money -= cost
        self.compass += 1
        return 0.2

    def _act_buy_info(self) -> float:
        if self.node_type(self.pos) != "CAMP":
            return -10.0
        if self.food_count() <= 0 or self.water_count() <= 0:
            return -10.0
        # Cost is time + 1 food + 1 water; time is accounted per step
        self.food_units.pop()
        self.water_units.pop()
        return 0.0

    def _act_rest(self) -> float:
        # No specific bonus other than consuming the day
        return 0.0

    def _act_mine(self) -> float:
        if self.pos != "MINE":
            return -10.0
        self.gold += 50
        return 10.0

    # ------------------------------------------------------------------
    # Weather + consumption
    # ------------------------------------------------------------------
    def _handle_weather(self, weather: int, node_type: str) -> Tuple[Optional[str], bool]:
        """Return (tool_used, lock_added)."""

        if self.locked_days > 0:
            return None, False

        if node_type in ["CAMP", "L"]:
            return None, False

        if weather in [self.SANDSTORM, self.STORM_HEAT]:
            if self.tent_uses > 0:
                self.tent_uses -= 1
                self.tent_weight = max(0, self.tent_weight - 20)
                return "tent", False
            if self.compass > 0:
                self.compass -= 1
                return "compass", False
            # locked
            self.locked_days = 3
            self.lock_weather = weather
            return None, True
        return None, False

    @staticmethod
    def _consume_units(units: list, amount: int) -> None:
        for _ in range(amount):
            if units:
                units.pop()
            else:
                break

    def _apply_consumption(self, weather: int, node_type: str, tool_used: Optional[str]) -> float:
        # Oasis: no water cost for the day and cannot die from water
        if node_type == "L":
            food_cost = 1
            water_cost = 0
        else:
            if weather == self.NORMAL:
                food_cost, water_cost = 1, 1
            elif weather == self.HEAT:
                food_cost, water_cost = 1, 3
            elif weather == self.SANDSTORM:
                if tool_used == "tent":
                    food_cost, water_cost = 1, 1
                else:
                    food_cost, water_cost = 5, 2
            else:  # STORM_HEAT
                if tool_used == "tent":
                    food_cost, water_cost = 1, 3
                else:
                    food_cost, water_cost = 5, 4

        self._consume_units(self.food_units, food_cost)
        self._consume_units(self.water_units, water_cost)

        return 0.0

    def _is_dead(self, node_type: str) -> bool:
        food_empty = self.food_count() == 0
        water_empty = self.water_count() == 0
        if node_type == "L" and water_empty:
            # Oasis prevents water death for the day
            return food_empty
        return food_empty or water_empty

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _obs(self) -> Dict[str, np.ndarray]:
        return {
            "day": np.array([self.day_count], dtype=np.int32),
            "node": np.array([self.node_to_idx[self.pos]], dtype=np.int32),
            "food": np.array([self.food_count()], dtype=np.int32),
            "water": np.array([self.water_count()], dtype=np.int32),
            "weight": np.array([self.compute_weight()], dtype=np.int32),
            "money": np.array([self.money], dtype=np.int32),
            "gold": np.array([self.gold], dtype=np.int32),
            "tent": np.array([self.tent_uses], dtype=np.int32),
            "compass": np.array([self.compass], dtype=np.int32),
            "locked": np.array([self.locked_days], dtype=np.int32),
            "weather": np.array([self._current_weather()], dtype=np.int32),
            "phase": np.array([self.phase], dtype=np.int32),
        }

