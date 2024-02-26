import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        return self.fc2(x)
    

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.memory)
    

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(action_size, 10000, 20)
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state, eps=0.01):
        if np.random.rand() <= eps:
            return random.randrange(self.action_size)
        #state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        action_values = self.model(state)
        return np.argmax(action_values.detach().numpy())

    def learn(self):
        if len(self.memory) < self.memory.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        Q_expected = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_targets_next = self.model(next_states).detach().max(1)[0]
        Q_targets = rewards + (0.99 * Q_targets_next * (1 - dones))
        loss = self.criterion(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, n_episodes=1000, max_t=1000):
        scores = []  
        for i_episode in range(1, n_episodes+1):
            state, info = env.reset()  
            score = 0
            for t in range(max_t):
                action = self.act(state)
                next_state, reward, done, truncated, info = env.step(action)  
                self.memory.add(state, action, reward, next_state, done or truncated)  
                state = next_state  
                score += reward
                self.learn()
                if done or truncated:
                    break
            scores.append(score)
            print(f"Episode {i_episode}/{n_episodes}, Score: {score}")
        return scores

# Initialize the Gym environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize DQN Agent
agent = DQNAgent(state_size, action_size)
scores = agent.train(env)  # Train the agent

env.close()
