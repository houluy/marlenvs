"""
Switch Riddle — 经典DQN实现
全连接网络 + 经验回放 + 目标网络
参数共享：所有智能体共用同一个策略网络
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
from switch_riddle.manual_control import ClassicStrategy


# ============================================================
# Q 网络：全连接神经网络
# ============================================================
class QNetwork(nn.Module):
    """全连接网络，输入观测向量，输出每个动作的 Q 值"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================
# 经验回放缓冲区
# ============================================================
class ReplayBuffer:
    """固定容量的经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================
# DQN 训练器
# ============================================================
class DQNTrainer:
    """管理 Q 网络 / 目标网络 / ε-greedy / 经验回放 / 梯度更新"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        success_frac: float = 0.5,
        device: Optional[str] = None,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        self.success_frac = success_frac

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.q_network = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss，比 MSE 更稳定

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.success_buffer = ReplayBuffer(buffer_capacity // 5)  # 成功经验池
        self.train_step_count = 0

    def select_action(self, obs: np.ndarray, epsilon: Optional[float] = None) -> int:
        """ε-greedy 动作选择"""
        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            return int(q_values.argmax(dim=1).item())

    def train_step(self) -> Optional[float]:
        """从两个回放缓冲区采样（成功池+主池），执行一步梯度更新"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        n_success = min(int(self.batch_size * self.success_frac), len(self.success_buffer))
        n_main = self.batch_size - n_success

        obs_list = []
        act_list = []
        rew_list = []
        nobs_list = []
        done_list = []

        if n_success > 0:
            o, a, r, no, dn = self.success_buffer.sample(n_success)
            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r)
            nobs_list.append(no)
            done_list.append(dn)

        o, a, r, no, dn = self.replay_buffer.sample(n_main)
        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        nobs_list.append(no)
        done_list.append(dn)

        obs = torch.cat(obs_list).to(self.device)
        actions = torch.cat(act_list).to(self.device)
        rewards = torch.cat(rew_list).to(self.device)
        next_obs = torch.cat(nobs_list).to(self.device)
        dones = torch.cat(done_list).to(self.device)

        q_values = self.q_network(obs)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_obs)
            max_next_q = next_q_values.max(dim=1).values
            target = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1

        # 硬更新目标网络
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """指数衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """保存模型 checkpoint"""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count,
            },
            path,
        )

    def load(self, path: str):
        """加载模型 checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.train_step_count = checkpoint.get("train_step_count", 0)


# ============================================================
# 训练 & 评估
# ============================================================

def _build_actions(env: SwitchRiddleEnv, active_agent: str, action: int) -> Dict[str, int]:
    """为所有智能体构建动作字典（非活跃智能体填 NOOP=1）"""
    return {agent: action if agent == active_agent else 1 for agent in env.possible_agents}


def train_episode(env: SwitchRiddleEnv, trainer: DQNTrainer) -> float:
    """训练一个 episode，返回团队总奖励。成功则加入成功经验池"""
    observations, _ = env.reset()
    episode_reward = 0.0
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    episode_transitions = []

    while env.agents:
        active_agent = env._active_agent
        obs = observations[active_agent]

        action = trainer.select_action(obs)
        actions = _build_actions(env, active_agent, action)

        next_observations, rewards, terms, truncs, _ = env.step(actions)

        done = any(terms.values()) or any(truncs.values())
        reward = rewards[active_agent]
        episode_reward += reward

        if done:
            next_obs = np.zeros(obs_dim, dtype=np.float32)
        else:
            next_obs = next_observations[env._active_agent]

        transition = (obs, action, reward, next_obs, float(done))
        trainer.replay_buffer.push(*transition)
        episode_transitions.append(transition)
        trainer.train_step()

        observations = next_observations

    # 成功 episode 的经验额外存入成功池
    if episode_reward > 0:
        for t in episode_transitions:
            trainer.success_buffer.push(*t)

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
        observations, _ = env.reset()
        episode_reward = 0.0
        steps = 0

        while env.agents:
            active_agent = env._active_agent
            obs = observations[active_agent]

            action = trainer.select_action(obs, epsilon=0.0)
            actions = _build_actions(env, active_agent, action)

            observations, rewards, terms, truncs, _ = env.step(actions)
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

    observations, _ = env.reset()
    episode_reward = 0.0

    while env.agents:
        env.render()

        active_agent = env._active_agent
        obs = observations[active_agent]

        action = trainer.select_action(obs, epsilon=0.0)
        actions = _build_actions(env, active_agent, action)

        observations, rewards, terms, truncs, _ = env.step(actions)
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
# 预热：经典策略采样
# ============================================================
def warmup_episode(env: SwitchRiddleEnv, trainer: DQNTrainer) -> float:
    """使用经典计数策略收集经验，不训练，成功则加入成功经验池"""
    observations, _ = env.reset()
    strategy = ClassicStrategy(env.n_agents)
    episode_reward = 0.0
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    episode_transitions = []

    while env.agents:
        active_agent = env._active_agent
        obs = observations[active_agent]
        agent_idx = env.idx_map[active_agent]

        action = strategy.act(agent_idx, env._switch)
        actions = _build_actions(env, active_agent, action)

        next_observations, rewards, terms, truncs, _ = env.step(actions)

        done = any(terms.values()) or any(truncs.values())
        reward = rewards[active_agent]
        episode_reward += reward

        if done:
            next_obs = np.zeros(obs_dim, dtype=np.float32)
        else:
            next_obs = next_observations[env._active_agent]

        transition = (obs, action, reward, next_obs, float(done))
        trainer.replay_buffer.push(*transition)
        episode_transitions.append(transition)

        observations = next_observations

    if episode_reward > 0:
        for t in episode_transitions:
            trainer.success_buffer.push(*t)

    return episode_reward


# ============================================================
# 主函数
# ============================================================
def main():
    # ---------- 超参数 ----------
    NUM_AGENTS = 3
    MAX_STEPS = 200
    NUM_EPISODES = 2000
    WARMUP_EPISODES = 200
    HIDDEN_DIMS = [128, 64]
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_CAPACITY = 50000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100
    SUCCESS_FRAC = 0.5
    EVAL_INTERVAL = 200

    SAVE_PATH = "checkpoints/switch_riddle_dqn.pt"

    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"智能体数量: {NUM_AGENTS}")

    train_env = SwitchRiddleEnv(num_agents=NUM_AGENTS, max_steps=MAX_STEPS)
    eval_env = SwitchRiddleEnv(num_agents=NUM_AGENTS, max_steps=MAX_STEPS)

    obs_dim = train_env.observation_spaces[train_env.possible_agents[0]].shape[0]
    action_dim = train_env.action_spaces[train_env.possible_agents[0]].n
    print(f"观测维度: {obs_dim}  动作数: {action_dim}")

    trainer = DQNTrainer(
        obs_dim=obs_dim,
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
        success_frac=SUCCESS_FRAC,
    )

    best_success_rate = 0.0
    episode_rewards: List[float] = []

    import datetime
    log_dir = f"runs/switch_riddle_dqn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志目录: {log_dir}")

    # ---------- 预热阶段：经典计数策略采样 ----------
    print(f"\n经典策略预热 ({WARMUP_EPISODES} episodes)...")
    warmup_success = 0
    for ep in range(1, WARMUP_EPISODES + 1):
        r = warmup_episode(train_env, trainer)
        if r > 0:
            warmup_success += 1
        if ep % 50 == 0:
            print(
                f"  预热 Ep {ep:4d} | "
                f"累计成功 {warmup_success} | "
                f"主池 {len(trainer.replay_buffer)} | "
                f"成功池 {len(trainer.success_buffer)}"
            )
    print(f"预热完成: 成功率 {warmup_success / WARMUP_EPISODES:.1%} | "
          f"主池 {len(trainer.replay_buffer)} | 成功池 {len(trainer.success_buffer)}")

    print(f"\n开始 DQN 训练 ({NUM_EPISODES} episodes)...\n")

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
            writer.add_scalar("buffer/main_size", len(trainer.replay_buffer), episode)
            writer.add_scalar("buffer/success_size", len(trainer.success_buffer), episode)

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

    writer.close()

    trainer.save(SAVE_PATH)
    print(f"\n模型已保存至 {SAVE_PATH}")

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
