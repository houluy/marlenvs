import functools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


def _env():
    from predator_prey.env import PredatorPreyEnv
    return PredatorPreyEnv()


class PredatorPreyEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "predator_prey_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        grid_size: int = 10,
        num_predators: int = 3,
        num_prey: int = 1,
        max_steps: int = 200,
        catch_radius: int = 1,
        render_mode: Optional[str] = None,
    ):
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.max_steps = max_steps
        self.catch_radius = catch_radius
        self.render_mode = render_mode

        self.predator_ids = [f"predator_{i}" for i in range(num_predators)]
        self.prey_ids = [f"prey_{i}" for i in range(num_prey)]
        self.possible_agents = self.predator_ids + self.prey_ids
        self.agents = self.possible_agents[:]

        self._action_space = Discrete(5)
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}

        obs_dim = 2 + 2 * num_predators + 2 * num_prey + 4
        self._observation_space = Box(
            low=-self.grid_size, high=self.grid_size,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._cell_size: int = 50

    def _obs(self, agent: str) -> np.ndarray:
        agent_idx = self.possible_agents.index(agent)
        agent_pos = self._pos[self.agent_name_to_idx[agent]]

        obs_list = [agent_pos[0] / self.grid_size, agent_pos[1] / self.grid_size]

        for pid in self.predator_ids:
            p_idx = self.agent_name_to_idx[pid]
            p_pos = self._pos[p_idx]
            dx = (p_pos[0] - agent_pos[0]) / self.grid_size
            dy = (p_pos[1] - agent_pos[1]) / self.grid_size
            obs_list.extend([dx, dy])

        for pid in self.prey_ids:
            p_idx = self.agent_name_to_idx[pid]
            p_pos = self._pos[p_idx]
            dx = (p_pos[0] - agent_pos[0]) / self.grid_size
            dy = (p_pos[1] - agent_pos[1]) / self.grid_size
            obs_list.extend([dx, dy])

        can_left = float(agent_pos[0] > 0 or self.grid_size > 0)
        can_right = float(agent_pos[0] < self.grid_size - 1)
        can_up = float(agent_pos[1] > 0)
        can_down = float(agent_pos[1] < self.grid_size - 1)
        obs_list.extend([can_left, can_right, can_up, can_down])

        return np.array(obs_list, dtype=np.float32)

    def _move(self, pos: np.ndarray, action: int) -> np.ndarray:
        if action == 0:
            return pos.copy()
        x, y = pos
        if action == 1:
            y = min(y + 1, self.grid_size - 1)
        elif action == 2:
            y = max(y - 1, 0)
        elif action == 3:
            x = max(x - 1, 0)
        elif action == 4:
            x = min(x + 1, self.grid_size - 1)
        return np.array([x, y])

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
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        total_agents = self.num_predators + self.num_prey
        self.agent_name_to_idx = {
            name: idx for idx, name in enumerate(self.possible_agents)
        }

        occupied = set()
        self._pos = np.zeros((total_agents, 2), dtype=np.int32)
        for i in range(total_agents):
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    self._pos[i] = [x, y]
                    break

        observations = {agent: self._obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _check_caught(self) -> Dict[str, bool]:
        caught = {}
        for prey_id in self.prey_ids:
            prey_idx = self.agent_name_to_idx[prey_id]
            prey_pos = self._pos[prey_idx]
            predators_on_prey = 0
            for pred_id in self.predator_ids:
                pred_idx = self.agent_name_to_idx[pred_id]
                pred_pos = self._pos[pred_idx]
                dist = np.abs(pred_pos - prey_pos).sum()
                if dist <= self.catch_radius:
                    predators_on_prey += 1
            caught[prey_id] = predators_on_prey >= 2
        return caught

    def step(self, actions: Dict[str, int]):
        if any(self.terminations.values()) or any(self.truncations.values()):
            return (
                {agent: self._obs(agent) for agent in self.agents},
                {agent: 0.0 for agent in self.agents},
                self.terminations.copy(),
                self.truncations.copy(),
                {agent: {} for agent in self.agents},
            )

        for agent, action in actions.items():
            idx = self.agent_name_to_idx[agent]
            self._pos[idx] = self._move(self._pos[idx], action)

        self._step_count += 1

        caught = self._check_caught()
        rewards = {}

        for agent in self.agents:
            if agent in self.predator_ids:
                rewards[agent] = -0.05
            else:
                rewards[agent] = 0.1

        for prey_id, is_caught in caught.items():
            if is_caught:
                prey_idx = self.agent_name_to_idx[prey_id]
                prey_pos = self._pos[prey_idx]
                for pred_id in self.predator_ids:
                    pred_idx = self.agent_name_to_idx[pred_id]
                    pred_pos = self._pos[pred_idx]
                    dist = np.abs(pred_pos - prey_pos).sum()
                    if dist <= self.catch_radius:
                        rewards[pred_id] += 10.0
                rewards[prey_id] = -10.0

        if any(caught.values()):
            self.terminations = {agent: True for agent in self.agents}

        if self._step_count >= self.max_steps:
            self.truncations = {agent: True for agent in self.agents}

        if all(self.terminations.values()) or all(self.truncations.values()):
            self.agents = []

        observations = {agent: self._obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, self.terminations, self.truncations, infos

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self._screen is None:
            pygame.init()
            size = self.grid_size * self._cell_size
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((size, size))
                pygame.display.set_caption("Predator-Prey")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((size, size))

        self._screen.fill((255, 255, 255))

        for x in range(self.grid_size + 1):
            pos = x * self._cell_size
            pygame.draw.line(self._screen, (200, 200, 200), (pos, 0), (pos, self.grid_size * self._cell_size))
            pygame.draw.line(self._screen, (200, 200, 200), (0, pos), (self.grid_size * self._cell_size, pos))

        for i, prey_id in enumerate(self.prey_ids):
            p_idx = self.agent_name_to_idx[prey_id]
            x, y = self._pos[p_idx]
            cx = x * self._cell_size + self._cell_size // 2
            cy = y * self._cell_size + self._cell_size // 2
            r = self._cell_size // 2 - 4
            pygame.draw.circle(self._screen, (0, 180, 0), (cx, cy), r)
            pygame.draw.circle(self._screen, (0, 100, 0), (cx, cy), r, 2)

        for i, pred_id in enumerate(self.predator_ids):
            p_idx = self.agent_name_to_idx[pred_id]
            x, y = self._pos[p_idx]
            cx = x * self._cell_size + self._cell_size // 2
            cy = y * self._cell_size + self._cell_size // 2
            r = self._cell_size // 2 - 4
            pygame.draw.circle(self._screen, (220, 50, 50), (cx, cy), r)
            pygame.draw.circle(self._screen, (150, 20, 20), (cx, cy), r, 2)

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
            pygame.display.flip()
            if self._clock:
                self._clock.tick(30)

            return None
        else:
            arr = np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )
            return arr

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None
