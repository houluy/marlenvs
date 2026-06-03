import sys

import numpy as np
import pygame

from predator_prey.env import PredatorPreyEnv


def main():
    env = PredatorPreyEnv(
        grid_size=10,
        num_predators=2,
        num_prey=2,
        max_steps=500,
        render_mode="human",
    )

    obs, info = env.reset()
    running = True
    human_control = "predator_0"

    while running:
        env.render()

        actions = {}
        for agent in env.agents:
            if agent == human_control:
                actions[agent] = 0
            elif agent in env.predator_ids:
                actions[agent] = np.random.randint(0, 5)
            else:
                actions[agent] = np.random.randint(0, 5)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    actions[human_control] = 1
                elif event.key == pygame.K_s:
                    actions[human_control] = 2
                elif event.key == pygame.K_a:
                    actions[human_control] = 3
                elif event.key == pygame.K_d:
                    actions[human_control] = 4
                elif event.key == pygame.K_SPACE:
                    actions[human_control] = 0
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    continue

                if event.key in (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_SPACE):
                    obs, rewards, terms, truncs, infos = env.step(actions)
                    total_pred_reward = sum(rewards.get(a, 0) for a in env.predator_ids)
                    total_prey_reward = sum(rewards.get(a, 0) for a in env.prey_ids)
                    print(
                        f"Step: {env._step_count} | "
                        f"Pred reward: {total_pred_reward:.2f} | "
                        f"Prey reward: {total_prey_reward:.2f}"
                    )
                    if any(terms.values()) or any(truncs.values()):
                        print("Episode ended!")
                        obs, info = env.reset()

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
