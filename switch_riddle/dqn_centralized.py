"""
Switch Riddle — 集中式DQN实现
单个控制器观察全局状态 env.state()，决定活跃智能体的动作
对比策略：与参数共享 DQN (dqn.py) 对比集中式 vs 分散式决策的差异
"""

import random
import sys
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from switch_riddle.env import SwitchRiddleEnv

# 复用 dqn.py 的网络结构和训练器（仅观测来源不同）
from switch_riddle.dqn import QNetwork, ReplayBuffer, DQNTrainer


# ============================================================
# 训练 & 评估（集中式版本）
# ============================================================

def _build_actions(env: SwitchRiddleEnv, active_agent: str, action: int) -> Dict[str, int]:
    """为所有智能体构建动作字典（非活跃智能体填 NOOP=1）"""
    return {agent: action if agent == active_agent else 1 for agent in env.possible_agents}


def train_episode(env: SwitchRiddleEnv, trainer: DQNTrainer) -> float:
    """
    训练一个 episode，返回团队总奖励。
    集中式版本：使用 env.state() 作为全局观测。
    """
    env.reset()
    state_dim = len(env.state())
    episode_reward = 0.0

    while env.agents:
        active_agent = env._active_agent
        obs = env.state()  # 全局状态，而非个体观测

        action = trainer.select_action(obs)
        actions = _build_actions(env, active_agent, action)

        _, rewards, terms, truncs, _ = env.step(actions)

        done = any(terms.values()) or any(truncs.values())
        reward = rewards[active_agent]
        episode_reward += reward

        if done:
            next_obs = np.zeros(state_dim, dtype=np.float32)
        else:
            next_obs = env.state()

        trainer.replay_buffer.push(obs, action, reward, next_obs, float(done))
        trainer.train_step()

    trainer.decay_epsilon()
    return episode_reward


def evaluate(
    env: SwitchRiddleEnv,
    trainer: DQNTrainer,
    num_episodes: int = 100,
) -> Dict[str, float]:
    """评估当前策略（关闭探索），返回成功率 / 平均奖励 / 平均步数"""
    success = 0
    total_rewards: List[float] = []
    total_steps: List[int] = []

    for _ in range(num_episodes):
        env.reset()
        episode_reward = 0.0
        steps = 0

        while env.agents:
            active_agent = env._active_agent
            obs = env.state()

            action = trainer.select_action(obs, epsilon=0.0)
            actions = _build_actions(env, active_agent, action)

            _, rewards, terms, truncs, _ = env.step(actions)
            episode_reward += rewards[active_agent]
            steps += 1

            if rewards[active_agent] > 0:
                success += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    return {
        "success_rate": success / num_episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_steps": np.mean(total_steps),
    }


def run_demo(env: SwitchRiddleEnv, trainer: DQNTrainer) -> float:
    """使用训练好的模型跑一个 episode 并渲染"""
    import pygame

    env.reset()
    episode_reward = 0.0

    while env.agents:
        env.render()

        active_agent = env._active_agent
        obs = env.state()

        action = trainer.select_action(obs, epsilon=0.0)
        actions = _build_actions(env, active_agent, action)

        _, rewards, terms, truncs, _ = env.step(actions)
        episode_reward += rewards[active_agent]

        if env.render_mode == "human":
            time.sleep(0.3)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return episode_reward

        if any(terms.values()) or any(truncs.values()):
            env.render()
            time.sleep(1.0)
            break

    return episode_reward


# ============================================================
# 主函数
# ============================================================
def main():
    # ---------- 超参数 ----------
    NUM_AGENTS = 3
    MAX_STEPS = 200
    NUM_EPISODES = 2000
    HIDDEN_DIMS = [128, 64]
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_CAPACITY = 50000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100
    EVAL_INTERVAL = 200

    SAVE_PATH = "checkpoints/switch_riddle_dqn_centralized.pt"

    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"智能体数量: {NUM_AGENTS}")

    train_env = SwitchRiddleEnv(num_agents=NUM_AGENTS, max_steps=MAX_STEPS)
    train_env.reset()
    state_dim = len(train_env.state())
    train_env.close()
    train_env = SwitchRiddleEnv(num_agents=NUM_AGENTS, max_steps=MAX_STEPS)

    eval_env = SwitchRiddleEnv(num_agents=NUM_AGENTS, max_steps=MAX_STEPS)

    action_dim = train_env.action_spaces[train_env.possible_agents[0]].n
    print(f"全局状态维度: {state_dim}  动作数: {action_dim}")
    print(f"（对比：分散式观测维度 = {train_env.observation_spaces[train_env.possible_agents[0]].shape[0]}）")

    trainer = DQNTrainer(
        obs_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=HIDDEN_DIMS,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    best_success_rate = 0.0
    episode_rewards: List[float] = []

    import datetime
    log_dir = f"runs/switch_riddle_centralized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")
    print(f"\n开始训练 ({NUM_EPISODES} episodes)...\n")

    global_step = 0

    for episode in range(1, NUM_EPISODES + 1):
        reward = train_episode(train_env, trainer)
        episode_rewards.append(reward)

        writer.add_scalar("train/episode_reward", reward, episode)
        writer.add_scalar("train/epsilon", trainer.epsilon, episode)

        if episode % EVAL_INTERVAL == 0:
            stats = evaluate(eval_env, trainer, num_episodes=100)
            avg_reward = np.mean(episode_rewards[-EVAL_INTERVAL:])

            writer.add_scalar("eval/success_rate", stats["success_rate"], episode)
            writer.add_scalar("eval/avg_reward", stats["avg_reward"], episode)
            writer.add_scalar("eval/avg_steps", stats["avg_steps"], episode)
            writer.add_scalar("train/avg_reward_200", avg_reward, episode)

            print(
                f"Ep {episode:5d} | "
                f"训练平均奖励 {avg_reward:+.2f} | "
                f"评估成功率 {stats['success_rate']:.1%} | "
                f"评估奖励 {stats['avg_reward']:+.2f} | "
                f"评估步数 {stats['avg_steps']:.0f} | "
                f"ε={trainer.epsilon:.3f}"
            )

            if stats["success_rate"] > best_success_rate:
                best_success_rate = stats["success_rate"]
                trainer.save(SAVE_PATH)
                print(f"  -> 保存最佳模型（成功率 {best_success_rate:.1%}）")

    # ---------- 最终评估 ----------
    print("\n========== 最终评估 ==========")
    final_stats = evaluate(eval_env, trainer, num_episodes=500)
    print(f"成功率: {final_stats['success_rate']:.1%}")
    print(f"平均奖励: {final_stats['avg_reward']:+.2f}")
    print(f"平均步数: {final_stats['avg_steps']:.0f}")

    trainer.save(SAVE_PATH)
    print(f"\n模型已保存至 {SAVE_PATH}")

    writer.close()

    # ---------- 渲染演示 ----------
    print("\n========== 渲染演示 ==========")
    demo_env = SwitchRiddleEnv(
        num_agents=NUM_AGENTS, max_steps=MAX_STEPS, render_mode="human"
    )
    demo_reward = run_demo(demo_env, trainer)
    print(f"演示奖励: {demo_reward:+.1f}")
    demo_env.close()


if __name__ == "__main__":
    main()
