import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from models.dqn import DQNAgent
from models.actor_critic import train_actor_critic

def plot_results(dqn_scores, ac_scores, title="DQN vs Actor-Critic"):
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_scores, label='DQN')
    plt.plot(ac_scores, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    env_name = 'CartPole-v1'
    n_episodes = 2000

    # Initialize environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    # Train DQN agent
    dqn_agent = DQNAgent(state_size, action_size)
    dqn_scores = dqn_agent.train(env_name, n_episodes)

    # Train Actor-Critic
    ac_scores = train_actor_critic(env_name, state_size, action_size, n_episodes)

    plot_results(dqn_scores, ac_scores)

    # Window size
    N = 50

    # Average of last N episodes
    dqn_average_last_N = np.mean(dqn_scores[-N:])
    ac_average_last_N = np.mean(ac_scores[-N:])

    print(f"Average reward for the last {N} episodes for DQN: {dqn_average_last_N:.2f}")
    print(f"Average reward for the last {N} episodes for Actor-Critic: {ac_average_last_N:.2f}")

if __name__ == '__main__':
    main()