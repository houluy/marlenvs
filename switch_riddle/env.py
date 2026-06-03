import functools
from typing import Dict, List, Optional

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class SwitchRiddleEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "switch_riddle_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_agents: int = 3,
        max_steps: int = 200,
        initial_switch: int | str = 0,
        render_mode: Optional[str] = None,
    ):
        self.n_agents = num_agents
        self.max_steps = max_steps
        self._initial_switch = initial_switch
        self.render_mode = render_mode

        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.possible_agents = self.agent_ids[:]
        self.agents = self.possible_agents[:]

        self._action_space = Discrete(3)
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}

        obs_dim = 3 + num_agents
        self._observation_space = Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None

        self.idx_map = {name: i for i, name in enumerate(self.possible_agents)}

    def _onehot(self, idx: int) -> np.ndarray:
        v = np.zeros(self.n_agents, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _obs(self, agent: str) -> np.ndarray:
        idx = self.idx_map[agent]
        return np.concatenate([
            np.array([float(self._switch)], dtype=np.float32),
            np.array([float(self._visited[idx])], dtype=np.float32),
            self._onehot(idx),
            np.array([float(self._active_agent == agent)], dtype=np.float32),
        ])

    def state(self) -> np.ndarray:
        return np.concatenate([
            np.array([float(self._switch)], dtype=np.float32),
            self._visited.astype(np.float32),
            self._onehot(self._active_agent_idx),
        ])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self._step_count = 0
        if self._initial_switch == "random":
            self._switch = np.random.randint(0, 2)
        else:
            self._switch = int(self._initial_switch)
        self._initial_switch_val = self._switch
        self._visited = np.zeros(self.n_agents, dtype=bool)
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self._last_action: Dict[str, Optional[int]] = {agent: None for agent in self.agents}
        self._last_reward: Dict[str, float] = {agent: 0.0 for agent in self.agents}

        self._active_agent_idx = np.random.randint(0, self.n_agents)
        self._active_agent = self.possible_agents[self._active_agent_idx]
        self._visited[self._active_agent_idx] = True

        observations = {agent: self._obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        if all(self.terminations.values()) or all(self.truncations.values()):
            obs = {agent: self._obs(agent) for agent in self.agents}
            return obs, {a: 0 for a in self.agents}, self.terminations, self.truncations, {a: {} for a in self.agents}

        active_action = actions[self._active_agent]
        prev_active = self._active_agent

        rewards = {agent: 0.0 for agent in self.agents}
        done_flag = False

        if active_action == 0:  # TOGGLE
            self._switch = 1 - self._switch
            self._last_action[self._active_agent] = 0
        elif active_action == 1:  # NOOP
            self._last_action[self._active_agent] = 1
        elif active_action == 2:  # DECLARE
            self._last_action[self._active_agent] = 2
            done_flag = True
            if self._visited.all():
                rewards = {agent: float(self.n_agents) for agent in self.agents}
            else:
                rewards = {agent: -float(self.n_agents) for agent in self.agents}

        for agent in self.possible_agents:
            self._last_reward[agent] = rewards[agent]

        if done_flag:
            self.terminations = {agent: True for agent in self.agents}
        elif self._step_count >= self.max_steps:
            rewards = {agent: -float(self.n_agents) for agent in self.agents}
            self.truncations = {agent: True for agent in self.agents}

        if all(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []

        if not done_flag and self._step_count < self.max_steps:
            self._active_agent_idx = np.random.randint(0, self.n_agents)
            self._active_agent = self.possible_agents[self._active_agent_idx]
            self._visited[self._active_agent_idx] = True

        observations = {agent: self._obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, self.terminations, self.truncations, infos

    @staticmethod
    def _action_name(action: Optional[int]) -> str:
        names = {0: "TOGGLE", 1: "NOOP", 2: "DECLARE", None: "NONE"}
        return names.get(action, "?")

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self._screen is None:
            pygame.init()
            if self._font is None:
                self._font = pygame.font.SysFont("menlo", 18)
            card_w = 140
            card_h = 180
            cols = min(self.n_agents, 5)
            rows = (self.n_agents + cols - 1) // cols
            pad = 20
            header_h = 80
            width = cols * card_w + (cols + 1) * pad
            height = header_h + rows * card_h + (rows + 1) * pad
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Switch Riddle")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((width, height))

        self._screen.fill((30, 30, 30))
        w = self._screen.get_width()
        pad = 20
        header_h = 80

        pygame.draw.rect(self._screen, (20, 20, 20), (0, 0, w, header_h))

        switch_color = (0, 200, 0) if self._switch == 1 else (100, 100, 100)
        switch_text = "ON" if self._switch == 1 else "OFF"
        sw = self._font.render(f"Switch: {switch_text}", True, switch_color)
        self._screen.blit(sw, (pad, 10))

        step_text = self._font.render(
            f"Step: {self._step_count}   Active: {self._active_agent}", True, (200, 200, 200)
        )
        self._screen.blit(step_text, (pad, 40))

        reward_str = " | ".join(
            f"{a}: {self._last_reward.get(a, 0):+.1f}" for a in self.possible_agents
        )[:120]
        rw = self._font.render(reward_str, True, (150, 150, 150))
        self._screen.blit(rw, (pad, header_h - 24))

        card_w = 140
        card_h = 180
        cols = min(self.n_agents, 5)
        rows = (self.n_agents + cols - 1) // cols

        for i, ag in enumerate(self.possible_agents):
            row = i // cols
            col = i % cols
            cx = pad + col * (card_w + pad)
            cy = header_h + pad + row * (card_h + pad)

            is_active = ag == self._active_agent
            border_color = (255, 255, 0) if is_active else (80, 80, 80)
            bg_color = (50, 50, 50) if is_active else (40, 40, 40)
            pygame.draw.rect(self._screen, bg_color, (cx, cy, card_w, card_h))
            pygame.draw.rect(self._screen, border_color, (cx, cy, card_w, card_h), 2)

            name = self._font.render(ag, True, (255, 255, 255))
            self._screen.blit(name, (cx + 10, cy + 10))

            id_label = self._font.render(f"ID: {i}", True, (180, 180, 180))
            self._screen.blit(id_label, (cx + 10, cy + 40))

            visited_color = (0, 200, 0) if self._visited[i] else (200, 50, 50)
            visited_text = "Entered" if self._visited[i] else "Not yet"
            vt = self._font.render(visited_text, True, visited_color)
            self._screen.blit(vt, (cx + 10, cy + 70))

            last_act = self._last_action.get(ag)
            act_text = f"Last: {self._action_name(last_act)}"
            act_color = (200, 200, 200)
            at = self._font.render(act_text, True, act_color)
            self._screen.blit(at, (cx + 10, cy + 100))

            last_rw = self._last_reward.get(ag, 0)
            rw_color = (0, 200, 0) if last_rw > 0 else (200, 50, 50) if last_rw < 0 else (150, 150, 150)
            rw_text = f"Reward: {last_rw:+.1f}"
            rt = self._font.render(rw_text, True, rw_color)
            self._screen.blit(rt, (cx + 10, cy + 130))

        if self.render_mode == "human":
            pygame.display.flip()
            if self._clock:
                self._clock.tick(10)
            return None
        else:
            arr = np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )
            return arr

    def close(self):
        if self._screen is not None:
            self._screen = None
            self._clock = None
            self._font = None
