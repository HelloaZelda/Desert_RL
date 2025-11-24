import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random
import math

class DesertGoldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        edges_path="map_edges.json",
        nodes_path="map_nodes.json",
        seed=42,
        num_teams=1,
        max_days=30
    ):
        super().__init__()

        random.seed(seed)
        np.random.seed(seed)

        # Load map
        self.edges = json.load(open(edges_path))
        self.nodes = json.load(open(nodes_path))

        # Parameters
        self.num_teams = num_teams
        self.max_days = max_days
        self.max_weight = 1000

        # Weather state
        self.storm_days_left = 0   # remaining days frozen by sandstorm
        self.is_heat = False       # heat wave active?
        self.day_count = 1
        self.is_day = True  # toggles: day-night-day-night...

        # Duel system
        self.reject_count = {}    # reject_count[team][enemy] = num
        self.extra_move = {}      # extra_move[team] = 1 means next turn free move

        # Observation space (vector of size 18)
        # [day, is_day, pos, food, water, gold, money,
        #  hp, weight, has_compass, tent_uses,
        #  storm_days_left, is_heat,
        #  near_mine, near_tomb, node_type,
        #  neighbor_cnt, extra_move]
        self.observation_space = spaces.Box(
            low=0, high=2000, shape=(18,), dtype=np.float32
        )
kk
        self.action_space = spaces.Discrete(18)

        # team states
        self.reset_all_states()

        self.done = False

    def reset_all_states(self):
        self.team_pos = {}
        self.team_food = {}
        self.team_water = {}
        self.team_gold = {}
        self.team_money = {}
        self.team_hp = {}
        self.team_compass = {}
        self.team_tent_uses = {}
        self.team_tent_weight = {}
        self.team_alive = {}

        for t in range(self.num_teams):
            self.team_pos[t] = "CAMP"
            self.team_food[t] = 5
            self.team_water[t] = 5
            self.team_gold[t] = 0
            self.team_money[t] = 100
            self.team_hp[t] = 10
            self.team_compass[t] = 0
            self.team_tent_uses[t] = 0
            self.team_tent_weight[t] = 0
            self.team_alive[t] = True

            self.reject_count[t] = {}
            self.extra_move[t] = 0

        self.day_count = 1
        self.is_day = True
        self.storm_days_left = 0
        self.is_heat = False
    def node_type(self, node):
        t = self.nodes.get(node, {}).get("type", "S")
        return t

    def step_weather(self):
        """Check if new day triggers weather (only desert S)."""
        if self.is_day:
            # 10% chance sandstorm
            if random.random() < 0.10:
                self.storm_days_left = 3
            else:
                self.storm_days_left = max(0, self.storm_days_left - 1)

            # heatwave: deserts always hot
            self.is_heat = True

    def is_sandstorm(self, node):
        return (self.node_type(node) == "S") and (self.storm_days_left > 0)

    def compute_weight(self, t):
        w = (
            self.team_food[t] * 10
            + self.team_water[t] * 25
            + self.team_gold[t] * 2
            + self.team_compass[t] * 10
        )
        if self.team_tent_uses[t] > 0:
            w += self.team_tent_weight[t]
        return w

    def check_weight(self, t):
        while self.compute_weight(t) > self.max_weight:
            if self.team_gold[t] > 0:
                self.team_gold[t] -= 1
            else:
                break
    def step(self, action, team=0):
        """Run one step for team 0 (single-agent training)."""

        if not self.team_alive[team]:
            return self._obs(team), -10, True, False, {}

        if self.done:
            return self._obs(team), 0, True, False, {}

        node = self.team_pos[team]
        node_t = self.node_type(node)

        # ---------- Freeze by Sandstorm ----------
        if self.is_sandstorm(node) and self.team_compass[team] == 0 and self.team_tent_uses[team] == 0:
            # stuck for 3 days: storm_days_left counts down daily
            if self.storm_days_left > 0:
                reward = -2
                self._advance_time()
                return self._obs(team), reward, False, False, {}

        # ---------- Extra Move ----------
        free_move = (self.extra_move[team] > 0)

        # ---------- Process Action ----------
        reward = 0

        # 0~7 Move
        if 0 <= action <= 7:
            reward += self._act_move(team, action)

        elif action == 8:   # mine
            reward += self._act_mine(team)

        elif action == 9:   # rest
            reward += self._act_rest(team)

        elif action == 10:  # buy food
            reward += self._act_buy_food(team)

        elif action == 11:  # buy water
            reward += self._act_buy_water(team)

        elif action == 12:  # buy tent
            reward += self._act_buy_tent(team)

        elif action == 13:  # buy compass
            reward += self._act_buy_compass(team)

        elif action == 14:  # duel request
            reward += self._act_duel_request(team)

        elif action == 15:  # accept duel
            reward += self._act_accept_duel(team)

        elif action == 16:  # reject duel
            reward += self._act_reject_duel(team)

        elif action == 17:  # return camp
            reward += self._act_return_camp(team)

        else:
            reward -= 1

        # if extra move was consumed:
        if free_move:
            self.extra_move[team] = 0

        # ---------- Resource Consumption ----------
        # Night/day consumption behavior:
        reward += self._apply_consumption(team)

        # ---------- Weight Check ----------
        self.check_weight(team)

        # ---------- Duel Aftermath: delayed death ---------- 
        if self.team_hp[team] == 1 and self.team_gold[team] == 0:
            # died next day
            self.team_hp[team] = 0
            self.team_alive[team] = False
            self.done = True
            return self._obs(team), -50, True, False, {}

        # ---------- End of Day Handling ----------
        if self._is_day_finished():
            self._advance_time()

        # ---------- Check End ----------
        terminated = False
        if not self.team_alive[team]:
            terminated = True
            self.done = True

        # reward shaping
        reward += self.team_gold[team] * 0.01

        return self._obs(team), reward, terminated, False, {}

    # ---------------------------------------------
    #             Action Implementations
    # ---------------------------------------------
    def _act_move(self, t, action):
        node = self.team_pos[t]
        neighbors = self.edges[node]

        if action >= len(neighbors):
            return -1  # invalid move

        target = neighbors[action]
        # Tent prevents sandstorm block
        if self.is_sandstorm(node) and self.team_tent_uses[t] == 0:
            if self.team_compass[t] == 0:
                # cannot move
                return -2

        self.team_pos[t] = target
        return 0.5

    def _act_mine(self, t):
        node = self.team_pos[t]
        if self.node_type(node) != "MINE":
            return -1
        # Mining: gold +25 (no pickaxe)
        self.team_gold[t] += 25
        return 1.0

    def _act_rest(self, t):
        # If tent available, reduce consumption
        if self.team_tent_uses[t] > 0:
            # tent use
            self.team_tent_uses[t] -= 1
            self.team_tent_weight[t] = max(0, self.team_tent_weight[t] - 20)
        return 0.2

    def _act_buy_food(self, t):
        if self.node_type(self.team_pos[t]) != "C":
            return -1
        if self.team_money[t] < 20:
            return -1
        self.team_money[t] -= 20
        self.team_food[t] += 1
        return 0.2

    def _act_buy_water(self, t):
        if self.node_type(self.team_pos[t]) != "C":
            return -1
        if self.team_money[t] < 20:
            return -1
        self.team_money[t] -= 20
        self.team_water[t] += 1
        return 0.2

    def _act_buy_tent(self, t):
        if self.node_type(self.team_pos[t]) != "C":
            return -1
        if self.team_money[t] < 400:
            return -1
        self.team_money[t] -= 400
        self.team_tent_uses[t] = 3
        self.team_tent_weight[t] = 70
        return 0.3

    def _act_buy_compass(self, t):
        if self.node_type(self.team_pos[t]) != "C":
            return -1
        if self.team_money[t] < 100:
            return -1
        self.team_money[t] -= 100
        self.team_compass[t] += 1
        return 0.2
    # ---------------------------------------------
    #                 Duel Logic
    # ---------------------------------------------
    def _act_duel_request(self, t):
        node = self.team_pos[t]

        # find other teams at same location
        others = [x for x in range(self.num_teams)
                  if x != t and self.team_alive[x] and self.team_pos[x] == node]

        if len(others) == 0:
            return -1

        target = others[0]  # choose first enemy (single-agent)

        # if in TOMB → forced duel
        if self.node_type(node) == "TOMB":
            return self._do_duel(t, target, forced=True)

        # normal duel: target may accept or reject
        if target not in self.reject_count[t]:
            self.reject_count[t][target] = 0

        # if rejected twice → forced duel
        if self.reject_count[t][target] >= 2:
            return self._do_duel(t, target, forced=True)

        # otherwise: duel request pending (AI must call accept/reject)
        self.pending_duel = (t, target)
        return 0.1

    def _act_accept_duel(self, t):
        if not hasattr(self, "pending_duel"):
            return -1
        attacker, defender = self.pending_duel
        if defender != t:
            return -1

        return self._do_duel(attacker, defender, forced=False)

    def _act_reject_duel(self, t):
        if not hasattr(self, "pending_duel"):
            return -1
        attacker, defender = self.pending_duel
        if defender != t:
            return -1

        # increment reject count
        if attacker not in self.reject_count:
            self.reject_count[attacker] = {}
        if t not in self.reject_count[attacker]:
            self.reject_count[attacker][t] = 0

        self.reject_count[attacker][t] += 1
        del self.pending_duel

        return 0.1

    # ---------------------------------------------------
    #               Duel Outcome
    # ---------------------------------------------------
    def _do_duel(self, attacker, defender, forced=False):
        # delete pending duel
        if hasattr(self, "pending_duel"):
            del self.pending_duel

        # compute power
        atk_power = (
            self.team_food[attacker]
            + self.team_water[attacker]
            + self.team_gold[attacker] * 5
            + self.team_money[attacker] * 0.1
            + (10 if self.team_tent_uses[attacker] > 0 else 0)
        )
        def_power = (
            self.team_food[defender]
            + self.team_water[defender]
            + self.team_gold[defender] * 5
            + self.team_money[defender] * 0.1
            + (10 if self.team_tent_uses[defender] > 0 else 0)
        )
        if atk_power + def_power == 0:
            p = 0.5
        else:
            p = atk_power / (atk_power + def_power)

        atk_wins = (random.random() < p)

        node = self.team_pos[attacker]

        # TOMB rules: winner takes all, loser survives but 0 gold
        if self.node_type(node) == "TOMB":
            if atk_wins:
                self.team_food[attacker] += self.team_food[defender]
                self.team_water[attacker] += self.team_water[defender]
                self.team_gold[attacker] += self.team_gold[defender]
                self.team_money[attacker] += self.team_money[defender]
                # defender stripped
                self.team_food[defender] = 0
                self.team_water[defender] = 0
                self.team_gold[defender] = 0
                self.team_money[defender] = 0
            else:
                self.team_food[defender] += self.team_food[attacker]
                self.team_water[defender] += self.team_water[attacker]
                self.team_gold[defender] += self.team_gold[attacker]
                self.team_money[defender] += self.team_money[attacker]
                # attacker stripped
                self.team_food[attacker] = 0
                self.team_water[attacker] = 0
                self.team_gold[attacker] = 0
                self.team_money[attacker] = 0
            # no death in TOMB duel
            return 2 if atk_wins else -2

        # ------------------------------------
        #  Normal Duel
        # ------------------------------------
        if atk_wins:
            # attacker wins: defender loses resources, HP becomes 1 → dies tomorrow
            gain_gold = min(self.team_gold[defender], 50)
            self.team_gold[attacker] += gain_gold
            self.team_gold[defender] -= gain_gold

            self.team_food[defender] = 0
            self.team_water[defender] = 0
            self.team_money[defender] = 0

            self.team_hp[defender] = 1  # delayed death

            # forced duel option: if multiple teams, will check automatically
            return 3.0

        else:
            # attacker loses
            if forced:
                # forced duel loss = instant death
                self.team_alive[attacker] = False
                self.team_hp[attacker] = 0
                return -5

            # normal duel fail: attacker stripped and delayed death
            self.team_food[attacker] = 0
            self.team_water[attacker] = 0
            self.team_money[attacker] = 0
            self.team_hp[attacker] = 1

            return -3

    # ---------------------------------------------------
    #         Consumption Logic (Food / Water)
    # ---------------------------------------------------
    def _apply_consumption(self, t):
        if not self.team_alive[t]:
            return 0

        node = self.team_pos[t]
        node_t = self.node_type(node)

        is_heat = (node_t == "S") or (node_t == "L")
        # L: oasis = no water consumption at all
        if node_t == "L":
            w_cost = 0
            f_cost = 1  # minor food cost?
        else:
            # water cost
            w_cost = 1
            if is_heat:
                w_cost *= 2  # heat doubles water cost

            if self.is_sandstorm(node) and self.team_compass[t] == 0 and self.team_tent_uses[t] == 0:
                # sandstorm extra
                w_cost += 1

            # food cost
            f_cost = 1

        # tent use reduces consumption if night
        if not self.is_day and self.team_tent_uses[t] > 0:
            f_cost = 0
            w_cost = 0

        # apply
        self.team_food[t] = max(0, self.team_food[t] - f_cost)
        self.team_water[t] = max(0, self.team_water[t] - w_cost)

        # starving or dehydrated?
        if self.team_food[t] == 0 or self.team_water[t] == 0:
            self.team_hp[t] -= 1
            if self.team_hp[t] <= 0:
                self.team_alive[t] = False
                return -10

        return 0

    # ---------------------------------------------------
    #              Observation Vector
    # ---------------------------------------------------
    def _obs(self, t):
        node = self.team_pos[t]
        node_t = self.node_type(node)
        neighbors = self.edges[node]

        near_mine = 1 if any(self.node_type(x) == "MINE" for x in neighbors) else 0
        near_tomb = 1 if any(self.node_type(x) == "TOMB" for x in neighbors) else 0

        obs = np.array([
            self.day_count,
            1 if self.is_day else 0,
            hash(node) % 10000,
            self.team_food[t],
            self.team_water[t],
            self.team_gold[t],
            self.team_money[t],
            self.team_hp[t],
            self.compute_weight(t),
            self.team_compass[t],
            self.team_tent_uses[t],
            self.storm_days_left,
            1 if self.is_heat else 0,
            near_mine,
            near_tomb,
            0 if node_t == "S" else (1 if node_t == "C" else 2),
            len(neighbors),
            self.extra_move[t]
        ], dtype=np.float32)

        return obs

    # ---------------------------------------------------
    #       Time Advance: Day → Night → Next Day
    # ---------------------------------------------------
    def _is_day_finished(self):
        # each step = 1 action within a phase
        return True  # treat each step as ending a phase

    def _advance_time(self):
        if self.is_day:
            self.is_day = False
        else:
            self.is_day = True
            self.day_count += 1
            self.step_weather()

    # ---------------------------------------------------
    #                   Reset
    # ---------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_all_states()
        self.storm_days_left = 0
        self.is_day = True
        self.is_heat = False
        self.day_count = 1
        self.done = False
        return self._obs(0), {}
