from __future__ import annotations
import glob
import os
import time

import supersuit as ss
from stable_baselines3 import A2C

from pettingzoo.atari import boxing_v2

def train_atari_supersuit(steps: int = 10_000, seed: int | None = 0):
    """
    Trains a A2C model on the Boxing environment from PettingZoo's Atari collection,
    with preprocessing steps.

    Args:
        steps (int): The total number of timesteps for which to train the model.
        seed (int | None): An optional seed for environment randomization.
    """ 
    env_name = boxing_v2.env()
    env = boxing_v2.parallel_env()

    env.reset(seed=seed)

    print(f"Starting training on {env_name}.")

    # SuperSuit wrappers
    env = ss.color_reduction_v0(env, mode='B')  # Color reduction
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize for CNN
    env = ss.frame_stack_v1(env, 4)  # Frame stacking
    
    # Convert to VecEnv for Stable Baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    model = A2C(
        "CnnPolicy",
        env,
        verbose=3,
        learning_rate=1e-3
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    #print(f"Finished training on {env.metadata['name']}.")
    print(f"Finished training on {env_name}")

    env.close()

def eval_atari_supersuit(num_games: int = 100, render_mode: str | None = None):
    """
    Evaluates a trained A2C model on the Boxing environment from PettingZoo's Atari collection.
    
    Args:
        num_games (int): The number of games to play for evaluation.
        render_mode (str | None): The rendering mode. Use 'human' for visual output or None for no rendering.
    """
    raw_env = boxing_v2.parallel_env()
    env_name = "boxing_v2"

    print(f"\nStarting evaluation on {env_name} (num_games={num_games}, render_mode={render_mode})")

    # Apply the same preprocessing as during training
    env = ss.color_reduction_v0(raw_env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

    try:
        latest_policy = max(glob.glob(f"{env_name}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        return

    model = A2C.load(latest_policy)

    rewards = []

    for _ in range(num_games):
        obs = env.reset()
        cumulative_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            cumulative_reward += reward

            if render_mode == 'human':
                raw_env.render()

            if done[0]:
                break

        rewards.append(cumulative_reward)

    avg_reward = sum(rewards) / num_games
    print(f"Avg reward: {avg_reward}")
    env.close()

if __name__ == "__main__":
    train_atari_supersuit(steps=1_000_000, seed=42) # Comment when only evaluation is needed
    eval_atari_supersuit(num_games=10, render_mode=None)
    eval_atari_supersuit(num_games=2, render_mode="human")  # Watch the trained agent
