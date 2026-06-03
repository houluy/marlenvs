import sys
import time

import pygame

from switch_riddle.env import SwitchRiddleEnv


class ClassicStrategy:
    _ACTION_TOGGLE = 0
    _ACTION_NOOP = 1
    _ACTION_DECLARE = 2

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.toggle_count = 0
        self.has_toggled = {i: False for i in range(num_agents)}

    def act(self, agent_idx: int, switch: int) -> int:
        if agent_idx == 0:
            if switch == 1:
                self.toggle_count += 1
                return self._ACTION_TOGGLE
            elif self.toggle_count >= self.num_agents - 1 and switch == 0:
                return self._ACTION_DECLARE
            else:
                return self._ACTION_NOOP
        else:
            if switch == 0 and not self.has_toggled[agent_idx]:
                self.has_toggled[agent_idx] = True
                return self._ACTION_TOGGLE
            else:
                return self._ACTION_NOOP


class TwoRoundStrategy:
    _ACTION_TOGGLE = 0
    _ACTION_NOOP = 1
    _ACTION_DECLARE = 2

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.toggle_count = 0
        self.toggles_used = {i: 0 for i in range(num_agents)}

    def act(self, agent_idx: int, switch: int) -> int:
        if agent_idx == 0:
            if switch == 1:
                self.toggle_count += 1
                return self._ACTION_TOGGLE
            elif self.toggle_count >= 2 * (self.num_agents - 1) and switch == 0:
                return self._ACTION_DECLARE
            else:
                return self._ACTION_NOOP
        else:
            if switch == 0 and self.toggles_used[agent_idx] < 2:
                self.toggles_used[agent_idx] += 1
                return self._ACTION_TOGGLE
            else:
                return self._ACTION_NOOP


def do_step(env, strategy, actions, active):
    obs, rewards, terms, truncs, infos = env.step(actions)
    total_reward = sum(rewards.values())
    if any(terms.values()) or any(truncs.values()):
        corrected = total_reward > 0
        print(f"  >> Episode ended! Correct={corrected} | Total Reward: {total_reward:+.1f}")
        return strategy, True
    return strategy, False


def wait_for_key(env):
    while True:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return


def main():
    num_agents = 3
    initial_switch = "random"
    strategy_type = "two_round"

    env = SwitchRiddleEnv(
        num_agents=num_agents,
        max_steps=500,
        initial_switch=initial_switch,
        render_mode="human",
    )

    env.reset()
    running = True
    human_control = "agent_0"

    if strategy_type == "two_round":
        strategy = TwoRoundStrategy(num_agents)
        target = 2 * (num_agents - 1)
    else:
        strategy = ClassicStrategy(num_agents)
        target = num_agents - 1

    print(f"Strategy: {strategy_type} | Initial switch: {env._initial_switch_val} | Target count: {target}")

    while running:
        env.render()

        if not env.agents:
            continue

        active = env._active_agent
        actions = {}
        for agent in env.agents:
            if agent == active:
                if agent == human_control:
                    actions[agent] = 1
                else:
                    idx = env.idx_map[agent]
                    actions[agent] = strategy.act(idx, env._switch)
            else:
                actions[agent] = 1

        if active != human_control:
            time.sleep(0.3)
            strategy, done = do_step(env, strategy, actions, active)
            if done:
                wait_for_key(env)
                running = False
            continue

        step_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    actions[active] = 0
                    step_taken = True
                elif event.key == pygame.K_n:
                    actions[active] = 1
                    step_taken = True
                elif event.key == pygame.K_d:
                    actions[active] = 2
                    step_taken = True
                elif event.key == pygame.K_r:
                    env.reset()
                    if strategy_type == "two_round":
                        strategy = TwoRoundStrategy(num_agents)
                    else:
                        strategy = ClassicStrategy(num_agents)
                    continue

                if step_taken:
                    strategy, done = do_step(env, strategy, actions, active)
                    if done:
                        wait_for_key(env)
                        running = False

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
