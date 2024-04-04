from __future__ import annotations
import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO

from pettingzoo.atari import boxing_v2

def train_atari_supersuit(env_fn, steps: int = 10_000, seed: int | None = 0):
    """
    Trains a PPO model on the Boxing environment from PettingZoo's Atari collection,
    with preprocessing steps.

    Args:
        steps (int): The total number of timesteps for which to train the model.
        seed (int | None): An optional seed for environment randomization.
    """ 
    env_name = boxing_v2.env()
    env = env_fn.parallel_env()

    # SuperSuit wrappers
    env = ss.color_reduction_v0(env, mode='B')  # Color reduction
    env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize for CNN
    env = ss.frame_stack_v1(env, 4)  # Frame stacking
    
    env.reset(seed=seed)

    print(f"Starting training on {env_name}.")

    # Convert to VecEnv for Stable Baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    model = PPO(
        "CnnPolicy",
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env_name}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {env_name}")

    env.close()

def eval_atari_supersuit(env_fn, num_games: int = 100, render_mode: str | None = None):
    """
    Evaluates a trained PPO model on the Boxing environment from PettingZoo's Atari collection.
    
    Args:
        num_games (int): The number of games to play for evaluation.
        render_mode (str | None): The rendering mode. Use 'human' for visual output or None for no rendering.
    """
    env = env_fn.env(render_mode=render_mode)
    env_name = "boxing_v2"

    # Apply the same preprocessing as during training
    env = ss.color_reduction_v0(env, mode='B')
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)

    print(f"\nStarting evaluation on {env_name} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(glob.glob(f"{env_name}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        return

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            if termination or truncation:
                for a in env.agents:
                    rewards[a] += env.rewards[a]
                break
            else:
                act = model.predict(obs, deterministic=True)[0]
            
            env.step(act)
        
    env.close()
        
    avg_reward = sum(rewards.values()) / len(rewards.values())
    print(f"Avg reward: {avg_reward}")
    #env.close()
    return avg_reward

if __name__ == "__main__":
    env_fn = boxing_v2

    # env_kwargs = dict(

    # )
    train_atari_supersuit(env_fn, steps=1_000_000, seed=0) # Comment when only evaluation is needed
    #eval_atari_supersuit(env_fn, num_games=10, render_mode=None)
    eval_atari_supersuit(env_fn, num_games=2, render_mode="human")  # Watch the trained agent
