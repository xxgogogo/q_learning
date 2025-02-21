# train.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from config import *


class DQNNetwork(nn.Module):
    """深度Q网络模型"""

    def __init__(self, input_size, output_size):
        """
        初始化DQN网络
        Args:
            input_size: 输入状态维度
            output_size: 输出动作维度
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存储转换"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样批次数据"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_size, action_size):
        """
        初始化DQN智能体
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
        """
        self.state_size = state_size
        self.action_size = action_size

        # 创建Q网络和目标网络
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters())
        self.memory = ReplayBuffer(10000)  # 经验回放缓冲区大小

        # 训练参数
        self.batch_size = 64
        self.gamma = DISCOUNT_FACTOR  # 折扣因子
        self.epsilon = EPSILON  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10  # 目标网络更新频率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将网络移至适当设备
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def get_action(self, state):
        """
        根据epsilon-greedy策略选择动作
        Args:
            state: 当前状态
        Returns:
            选择的动作
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def train(self, batch_size):
        """
        训练网络
        Args:
            batch_size: 批次大小
        Returns:
            float: 损失值
        """
        if len(self.memory) < batch_size:
            return 0

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, episode):
        """
        保存模型
        Args:
            episode: 当前训练轮数
        """
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, os.path.join(PTH_DIR, f'model_episode_{episode}.pth'))

    def load_model(self, path):
        """
        加载模型
        Args:
            path: 模型文件路径
        Returns:
            int: 加载的模型的训练轮数
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            return checkpoint['episode']
        return 0


def create_trainer(state_size=4, action_size=5):
    """
    创建训练器
    Args:
        state_size: 状态空间大小
        action_size: 动作空间大小
    Returns:
        DQNAgent: 训练器实例
    """
    return DQNAgent(state_size, action_size)
