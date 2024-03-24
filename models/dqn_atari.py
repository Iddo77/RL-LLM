import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque

from models.game_info import GameInfo
from utils import preprocess_frame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        # Each state per env.step is a screen image that is resized to 84x84.
        # A stack of 4 frames is used, to be able to see motion. Therefore, the state shape is [batch_size, 4, 84, 84].
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)  # Adjust the size based on the output of conv2
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, buffer_size=100000, batch_size=32):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.current_size = 0
        self.max_size = buffer_size

    def add(self, state, action, reward, next_state, done):
        if self.current_size == self.max_size:
            # discard oldest transition
            self.memory.popleft()
            self.current_size -= 1
        self.memory.append((state, action, reward, next_state, done))
        self.current_size += 1

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
        dones = torch.tensor(dones, dtype=torch.float, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, action_size, learn_freq=4, target_update_freq=10000):
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.model = QNetwork(action_size).to(device)
        self.target_model = QNetwork(action_size).to(device)
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.eps = 1.0  # epsilon is for epsilon-greedy exploration
        self.eps_decay = 0.998  # it decays over time, so in the beginning more exploration and later more exploitation
        self.eps_min = 0.01
        self.gamma = 0.99  # the discount factor, that discounts the value of future states
        self.learn_freq = learn_freq  # how often to call self.learn()
        self.target_update_freq = target_update_freq  # how often to update the target model
        self.num_steps = 0

    def act(self, state_tensor):
        # Assume state_tensor is already a PyTorch tensor on the correct device
        # disable gradient calculation when acting
        with torch.no_grad():
            action_values = self.model(state_tensor)
        # epsilon greedy action selection to balance exploration vs exploitation
        if np.random.rand() <= self.eps:
            # random action with probability eps (exploration)
            return random.randrange(self.action_size)
        # best action (exploitation)
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self):
        if len(self.memory) < self.memory.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample()

        # update both the model and the target_model (Double DQN)
        next_actions = self.model(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.target_model(next_states).detach().gather(1, next_actions).squeeze(1)

        # compute Q targets for current states using Bellman equation
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # get expected Q values from local model
        Q_expected = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # calculate loss and do gradient descent
        loss = self.criterion(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env,  game_info: GameInfo, n_episodes=10000, max_t=1000, save_interval=100, log_interval=10,
              weights_dir='DQN/weights'):
        scores = []
        eps_history = []  # To keep track of epsilon over time
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)

        for i_episode in range(1, n_episodes + 1):
            # Reset the environment and preprocess the initial state
            raw_state, info = env.reset()
            # The raw state is a screen-dump of the game.
            # It is resized to 84x84 and turned to grayscale during preprocessing.
            state = preprocess_frame(raw_state, game_info.crop_values)
            state = np.stack([state] * 4, axis=0)  # Stack the initial state 4 times

            score = 0
            for t in range(max_t):
                # place state on device
                state_tensor = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
                # select an action
                action = self.act(state_tensor)
                # execute the action and get the reward from the environment
                next_raw_state, reward, done, truncated, info = env.step(action)
                next_state_frame = preprocess_frame(next_raw_state, game_info.crop_values)
                # Update the state stack with the new frame
                next_state = np.append(state[1:, :, :], np.expand_dims(next_state_frame, 0), axis=0)
                # save state in memory
                self.memory.add(state, action, reward, next_state, done or truncated)

                state = next_state
                score += reward

                self.num_steps += 1
                if self.num_steps % self.learn_freq == 0:
                    self.learn()

                if self.num_steps % self.target_update_freq == 0:
                    # clone self.model and set as the new target model
                    self.target_model.load_state_dict(self.model.state_dict())
                    self.target_model.eval()

                if done or truncated:
                    # decay epsilon, so that more exploitation is done and less exploration
                    self.eps = max(self.eps_min, self.eps_decay * self.eps)
                    break

            scores.append(score)
            eps_history.append(self.eps)

            # Logging
            if i_episode % log_interval == 0:
                average_score = np.mean(scores[-log_interval:])
                print(f"Episode {i_episode}/{n_episodes} - Average Score: {average_score}, Epsilon: {self.eps}")

            # save weights when save interval is reached
            if i_episode % save_interval == 0:
                weights_path = os.path.join(weights_dir, f"weights_episode_{i_episode}.pth")
                self.save_weights(weights_path)
                print(f"Weights saved at '{weights_path}' after episode {i_episode}")

        # save final weights
        final_weights_path = os.path.join(weights_dir, "final_weights.pth")
        self.save_weights(final_weights_path)
        print(f"Final weights saved at '{final_weights_path}'")

        env.close()
        return scores, eps_history

    def load_weights(self, file_path):
        state = torch.load(file_path)
        self.model.load_state_dict(state["model_state"])
        self.target_model.load_state_dict(state["target_model_state"])
        self.model.to(device)
        self.target_model.to(device)

    def save_weights(self, file_path):
        state = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict()
        }
        torch.save(state, file_path)


if __name__ == '__main__':
    env = gym.make('BreakoutNoFrameskip-v4')
    agent = DQNAgent(env.action_space.n)
    scores, eps_history = agent.train(env, GameInfo.BREAKOUT)
    env.close()
