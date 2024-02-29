import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

        # self.actor = nn.Sequential(
        #     nn.Linear(state_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, action_size),
        #     nn.Softmax(dim=-1)
        # )

        # self.critic = nn.Sequential(
        #     nn.Linear(state_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )

    def forward(self, state):
        x = self.relu(self.fc1(state))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values
        
    # def forward(self, state):
    #     action_probs = self.actor(state)
    #     state_values = self.critic(state)
    #     return action_probs, state_values
    

def train_actor_critic(env_name, state_size, action_size, n_episodes=1000, gamma=0.99, lr=1e-3):
    env = gym.make(env_name)
    model = ActorCritic(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scores = []

    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = model(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, truncated, info = env.step(action.item())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_state_value = model(next_state)

            # Advantage
            advantage = reward + gamma * next_state_value * (1 - int(done)) - state_value

            # Entropy
            entropy = -torch.sum(action_probs * torch.log(action_probs))

            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)

            # Combine losses
            loss = actor_loss + critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            state = next_state
            total_reward += reward
            
            if done or truncated:
                break

        scores.append(total_reward)
        print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")

        # Learning rate scheduling
        if episode % 100 == 0 and episode > 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.9

    env.close()
    return scores

# Gym environment
# env = gym.make(env_name)            

# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n

# # Train Actor-Critic
# train_actor_critic(env, state_size, action_size)

# env.close()