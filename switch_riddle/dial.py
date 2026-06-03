"""
Switch Riddle — DIAL (Differentiable Inter-Agent Learning)
FC特征提取 → LSTM记忆 → Q值输出 + 消息输出
消息通过Gumbel-Softmax离散化，梯度可穿过消息传递到前一个智能体
"""

import random
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from switch_riddle.env import SwitchRiddleEnv
from switch_riddle.manual_control import ClassicStrategy


# ============================================================
# DIAL 网络：FC + LSTM + Q头 + 消息头
# ============================================================
class DRQNDialNetwork(nn.Module):
    def __init__(
        self,
        aug_obs_dim: int,
        action_dim: int,
        msg_vocab: int = 2,
        lstm_hidden: int = 128,
        fc_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if fc_dims is None:
            fc_dims = [128, 64]

        layers = []
        in_dim = aug_obs_dim
        for h_dim in fc_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.fc = nn.Sequential(*layers)

        self.lstm_hidden_size = lstm_hidden
        self.lstm = nn.LSTM(
            input_size=fc_dims[-1],
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.q_head = nn.Linear(lstm_hidden, action_dim)
        self.msg_head = nn.Linear(lstm_hidden, msg_vocab)
        self.msg_vocab = msg_vocab

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feat = self.fc(obs_seq)
        lstm_out, new_hidden = self.lstm(feat, hidden)
        q_values = self.q_head(lstm_out)
        msg_logits = self.msg_head(lstm_out)
        return q_values, msg_logits, new_hidden


# ============================================================
# 序列经验回放缓冲区（存储增强观测 + 隐状态）
# ============================================================
class SequenceReplayBuffer:
    """每个episode是一个list of (aug_obs, action, reward, aug_next_obs, done, h, c)"""

    def __init__(self, max_episodes: int = 1000, seq_len: int = 10, lstm_hidden: int = 128):
        self.buffer: deque = deque(maxlen=max_episodes)
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden

    def push_episode(self, transitions: List[Tuple]):
        if len(transitions) >= self.seq_len:
            self.buffer.append(transitions)

    def sample(self, batch_size: int, random_start: bool = True):
        obs_seqs, act_seqs, rew_seqs, nobs_seqs, done_seqs = [], [], [], [], []
        h0_list, c0_list = [], []

        zero_h = np.zeros((1, 1, self.lstm_hidden), dtype=np.float32)
        zero_c = np.zeros((1, 1, self.lstm_hidden), dtype=np.float32)

        for _ in range(batch_size):
            episode = random.choice(self.buffer)
            if random_start and len(episode) > self.seq_len:
                start = random.randint(0, len(episode) - self.seq_len)
            else:
                start = 0
            seq = episode[start:start + self.seq_len]

            if start > 0:
                _, _, _, _, _, h_prev, c_prev = episode[start - 1]
                h0_list.append(h_prev)
                c0_list.append(c_prev)
            else:
                h0_list.append(zero_h)
                c0_list.append(zero_c)

            o, a, r, no, d, *_ = zip(*seq)
            obs_seqs.append(np.array(o, dtype=np.float32))
            act_seqs.append(list(a))
            rew_seqs.append(list(r))
            nobs_seqs.append(np.array(no, dtype=np.float32))
            done_seqs.append(list(d))

        h0 = torch.FloatTensor(np.concatenate(h0_list, axis=1))
        c0 = torch.FloatTensor(np.concatenate(c0_list, axis=1))

        return (
            torch.FloatTensor(np.stack(obs_seqs)),
            torch.LongTensor(act_seqs),
            torch.FloatTensor(rew_seqs),
            torch.FloatTensor(np.stack(nobs_seqs)),
            torch.FloatTensor(done_seqs),
            h0, c0,
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================
# DIAL 训练器
# ============================================================
class DIALTrainer:
    def __init__(
        self,
        aug_obs_dim: int,
        action_dim: int,
        msg_vocab: int = 2,
        lstm_hidden: int = 128,
        fc_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        max_episodes: int = 1000,
        seq_len: int = 10,
        batch_size: int = 32,
        target_update_freq: int = 100,
        success_frac: float = 0.5,
        device: Optional[str] = None,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden
        self.success_frac = success_frac
        self.msg_vocab = msg_vocab

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_network = DRQNDialNetwork(
            aug_obs_dim, action_dim, msg_vocab, lstm_hidden, fc_dims
        ).to(self.device)
        self.target_network = DRQNDialNetwork(
            aug_obs_dim, action_dim, msg_vocab, lstm_hidden, fc_dims
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = SequenceReplayBuffer(max_episodes, seq_len, lstm_hidden)
        self.success_buffer = SequenceReplayBuffer(max_episodes // 5, seq_len, lstm_hidden)
        self.train_step_count = 0
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(self):
        self.hidden_state = None

    def _init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=self.device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=self.device)
        return (h, c)

    def _get_hidden_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.hidden_state is None:
            h = np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
            c = np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
        else:
            h = self.hidden_state[0].detach().cpu().numpy().copy()
            c = self.hidden_state[1].detach().cpu().numpy().copy()
        return h, c

    def _gen_message(
        self, msg_logits: torch.Tensor, training: bool, tau: float = 1.0
    ) -> torch.Tensor:
        """生成离散消息：训练时用Gumbel-Softmax，评估时用argmax"""
        if training:
            msg = F.gumbel_softmax(msg_logits, tau=tau, hard=True, dim=-1)
        else:
            idx = msg_logits.argmax(dim=-1, keepdim=True)
            msg = torch.zeros_like(msg_logits).scatter_(-1, idx, 1.0)
        return msg

    def select_action(
        self, aug_obs: np.ndarray, epsilon: Optional[float] = None, training: bool = True
    ) -> Tuple[int, np.ndarray]:
        obs_tensor = torch.FloatTensor(aug_obs).unsqueeze(0).unsqueeze(0).to(self.device)

        if self.hidden_state is None:
            self.hidden_state = self._init_hidden(1)

        eps = epsilon if epsilon is not None else self.epsilon
        if random.random() < eps:
            with torch.no_grad():
                _, msg_logits, self.hidden_state = self.q_network(obs_tensor, self.hidden_state)
            msg = self._gen_message(msg_logits.squeeze(0), training=False)
            return random.randint(0, self.action_dim - 1), msg.squeeze(0).cpu().numpy()

        with torch.no_grad():
            q_values, msg_logits, self.hidden_state = self.q_network(obs_tensor, self.hidden_state)
        action = int(q_values.squeeze().argmax().item())
        msg = self._gen_message(msg_logits.squeeze(0), training=training)
        return action, msg.squeeze(0).cpu().numpy()

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        n_success = min(int(self.batch_size * self.success_frac), len(self.success_buffer))
        n_main = self.batch_size - n_success

        obs_list, act_list, rew_list, nobs_list, done_list = [], [], [], [], []
        h0_list, c0_list = [], []

        if n_success > 0:
            o, a, r, no, dn, h0_s, c0_s = self.success_buffer.sample(n_success, random_start=False)
            obs_list.append(o)
            act_list.append(a)
            rew_list.append(r)
            nobs_list.append(no)
            done_list.append(dn)
            h0_list.append(h0_s)
            c0_list.append(c0_s)

        o, a, r, no, dn, h0_m, c0_m = self.replay_buffer.sample(n_main, random_start=False)
        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        nobs_list.append(no)
        done_list.append(dn)
        h0_list.append(h0_m)
        c0_list.append(c0_m)

        obs_seq = torch.cat(obs_list).to(self.device)
        act_seq = torch.cat(act_list).to(self.device)
        rew_seq = torch.cat(rew_list).to(self.device)
        nobs_seq = torch.cat(nobs_list).to(self.device)
        done_seq = torch.cat(done_list).to(self.device)
        h0 = torch.cat(h0_list, dim=1).to(self.device)
        c0 = torch.cat(c0_list, dim=1).to(self.device)

        batch = obs_seq.size(0)
        q_values, _, _ = self.q_network(obs_seq, (h0, c0))
        q_value = q_values.gather(2, act_seq.unsqueeze(2)).squeeze(2)

        with torch.no_grad():
            next_q_values, _, _ = self.target_network(nobs_seq, (h0, c0))
            max_next_q = next_q_values.max(dim=2).values
            target = rew_seq + self.gamma * max_next_q * (1.0 - done_seq)

        loss = self.loss_fn(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1

        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
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
    return {agent: action if agent == active_agent else 1 for agent in env.possible_agents}


def _collect_transition(
    aug_obs: np.ndarray,
    action: int,
    reward: float,
    aug_next_obs: np.ndarray,
    done: float,
    trainer: DIALTrainer,
) -> Tuple:
    h_np, c_np = trainer._get_hidden_numpy()
    return (aug_obs.copy(), action, reward, aug_next_obs.copy(), done, h_np, c_np)


def _zero_msg(msg_vocab: int) -> np.ndarray:
    return np.zeros(msg_vocab, dtype=np.float32)


def warmup_episode(env: SwitchRiddleEnv, trainer: DIALTrainer) -> float:
    """预热：ClassicStrategy提供动作，网络提供消息，填充序列缓冲"""
    observations, _ = env.reset()
    strategy = ClassicStrategy(env.n_agents)
    trainer.reset_hidden()
    episode_reward = 0.0
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    aug_obs_dim = obs_dim + trainer.msg_vocab
    episode_transitions: List[Tuple] = []
    prev_msg = _zero_msg(trainer.msg_vocab)

    while env.agents:
        active_agent = env._active_agent
        obs = observations[active_agent]
        agent_idx = env.idx_map[active_agent]

        expert_action = strategy.act(agent_idx, env._switch)
        aug_obs = np.concatenate([obs, prev_msg])

        _, msg = trainer.select_action(aug_obs, epsilon=0.0, training=False)

        actions = _build_actions(env, active_agent, expert_action)
        next_observations, rewards, terms, truncs, _ = env.step(actions)

        done = any(terms.values()) or any(truncs.values())
        reward = rewards[active_agent]
        episode_reward += reward

        if done:
            aug_next_obs = np.zeros(aug_obs_dim, dtype=np.float32)
        else:
            next_obs = next_observations[env._active_agent]
            aug_next_obs = np.concatenate([next_obs, msg])

        episode_transitions.append(
            _collect_transition(aug_obs, expert_action, reward, aug_next_obs, float(done), trainer)
        )
        observations = next_observations
        prev_msg = msg

    trainer.replay_buffer.push_episode(episode_transitions)
    if episode_reward > 0:
        trainer.success_buffer.push_episode(episode_transitions)
    return episode_reward


def train_episode(env: SwitchRiddleEnv, trainer: DIALTrainer) -> float:
    observations, _ = env.reset()
    trainer.reset_hidden()
    episode_reward = 0.0
    obs_dim = env.observation_spaces[env.possible_agents[0]].shape[0]
    aug_obs_dim = obs_dim + trainer.msg_vocab
    episode_transitions: List[Tuple] = []
    prev_msg = _zero_msg(trainer.msg_vocab)

    while env.agents:
        active_agent = env._active_agent
        obs = observations[active_agent]
        aug_obs = np.concatenate([obs, prev_msg])

        action, msg = trainer.select_action(aug_obs, training=True)
        actions = _build_actions(env, active_agent, action)

        next_observations, rewards, terms, truncs, _ = env.step(actions)

        done = any(terms.values()) or any(truncs.values())
        reward = rewards[active_agent]
        episode_reward += reward

        if done:
            aug_next_obs = np.zeros(aug_obs_dim, dtype=np.float32)
        else:
            next_obs = next_observations[env._active_agent]
            aug_next_obs = np.concatenate([next_obs, msg])

        episode_transitions.append(
            _collect_transition(aug_obs, action, reward, aug_next_obs, float(done), trainer)
        )
        observations = next_observations
        prev_msg = msg

    trainer.replay_buffer.push_episode(episode_transitions)
    if episode_reward > 0:
        trainer.success_buffer.push_episode(episode_transitions)

    trainer.train_step()
    trainer.decay_epsilon()
    return episode_reward


def evaluate(
    env: SwitchRiddleEnv,
    trainer: DIALTrainer,
    num_episodes: int = 100,
) -> Dict[str, float]:
    success = 0
    total_rewards: List[float] = []
    total_steps: List[int] = []
    msg_vocab = trainer.msg_vocab

    for _ in range(num_episodes):
        observations, _ = env.reset()
        trainer.reset_hidden()
        episode_reward = 0.0
        steps = 0
        prev_msg = _zero_msg(msg_vocab)

        while env.agents:
            active_agent = env._active_agent
            obs = observations[active_agent]
            aug_obs = np.concatenate([obs, prev_msg])

            action, msg = trainer.select_action(aug_obs, epsilon=0.0, training=False)
            actions = _build_actions(env, active_agent, action)

            observations, rewards, terms, truncs, _ = env.step(actions)
            episode_reward += rewards[active_agent]
            steps += 1
            prev_msg = msg

        if episode_reward > 0:
            success += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)

    return {
        "success_rate": success / num_episodes,
        "avg_reward": np.mean(total_rewards),
        "avg_steps": np.mean(total_steps),
    }


def run_demo(env: SwitchRiddleEnv, trainer: DIALTrainer) -> float:
    import pygame

    observations, _ = env.reset()
    trainer.reset_hidden()
    episode_reward = 0.0
    msg_vocab = trainer.msg_vocab
    prev_msg = _zero_msg(msg_vocab)

    while env.agents:
        env.render()

        active_agent = env._active_agent
        obs = observations[active_agent]
        aug_obs = np.concatenate([obs, prev_msg])

        action, msg = trainer.select_action(aug_obs, epsilon=0.0, training=False)
        actions = _build_actions(env, active_agent, action)

        observations, rewards, terms, truncs, _ = env.step(actions)
        episode_reward += rewards[active_agent]
        prev_msg = msg

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
    NUM_AGENTS = 3
    MAX_STEPS = 200
    WARMUP_EPISODES = 200
    NUM_EPISODES = 2000
    MSG_VOCAB = 2
    FC_DIMS = [128, 64]
    LSTM_HIDDEN = 128
    SEQ_LEN = 10
    LR = 1e-3
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    MAX_EPISODES = 1000
    BATCH_SIZE = 32
    TARGET_UPDATE_FREQ = 100
    SUCCESS_FRAC = 0.5
    EVAL_INTERVAL = 200

    SAVE_PATH = "checkpoints/switch_riddle_dial.pt"

    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"智能体: {NUM_AGENTS} | LSTM: {LSTM_HIDDEN} | 序列: {SEQ_LEN} | 消息词表: {MSG_VOCAB}")

    train_env = SwitchRiddleEnv(num_agents=NUM_AGENTS, max_steps=MAX_STEPS)
    eval_env = SwitchRiddleEnv(num_agents=NUM_AGENTS, max_steps=MAX_STEPS)

    obs_dim = train_env.observation_spaces[train_env.possible_agents[0]].shape[0]
    action_dim = train_env.action_spaces[train_env.possible_agents[0]].n
    aug_obs_dim = obs_dim + MSG_VOCAB
    print(f"观测维度: {obs_dim} + {MSG_VOCAB}消息 = {aug_obs_dim}  动作数: {action_dim}")

    trainer = DIALTrainer(
        aug_obs_dim=aug_obs_dim,
        action_dim=action_dim,
        msg_vocab=MSG_VOCAB,
        lstm_hidden=LSTM_HIDDEN,
        fc_dims=FC_DIMS,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        max_episodes=MAX_EPISODES,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        success_frac=SUCCESS_FRAC,
    )

    import datetime
    log_dir = f"runs/switch_riddle_dial_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {log_dir}")

    print(f"\n序列缓冲预热 ({WARMUP_EPISODES} episodes)...")
    warmup_success = 0
    for ep in range(1, WARMUP_EPISODES + 1):
        r = warmup_episode(train_env, trainer)
        if r > 0:
            warmup_success += 1
        if ep % 50 == 0:
            print(
                f"  预热 Ep {ep:4d} | 累计成功 {warmup_success} | "
                f"主池 {len(trainer.replay_buffer)} | 成功池 {len(trainer.success_buffer)}"
            )
    print(f"预热完成: 主池 {len(trainer.replay_buffer)} | 成功池 {len(trainer.success_buffer)}")

    print(f"\n开始 DIAL 训练 ({NUM_EPISODES} episodes)...\n")

    best_success_rate = 0.0
    episode_rewards: List[float] = []

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

    print("\n========== 最终评估 ==========")
    final_stats = evaluate(eval_env, trainer, num_episodes=500)
    print(f"成功率: {final_stats['success_rate']:.1%}")
    print(f"平均奖励: {final_stats['avg_reward']:+.2f}")
    print(f"平均步数: {final_stats['avg_steps']:.0f}")

    writer.close()
    trainer.save(SAVE_PATH)
    print(f"\n模型已保存至 {SAVE_PATH}")

    print("\n========== 渲染演示 ==========")
    demo_env = SwitchRiddleEnv(
        num_agents=NUM_AGENTS, max_steps=MAX_STEPS, render_mode="human"
    )
    demo_reward = run_demo(demo_env, trainer)
    print(f"演示奖励: {demo_reward:+.1f}")
    demo_env.close()


if __name__ == "__main__":
    main()
