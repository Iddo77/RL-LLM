from __future__ import annotations
import glob
import os
import time
import argparse

import supersuit as ss
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure

from pettingzoo.atari import boxing_v2

def train_atari_supersuit(model_type, env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    """
    Trains a model (A2C or PPO) on the Boxing environment from PettingZoo's Atari collection.

    Args:
        model_type (str): The type of RL model to use.
        steps (int): The total number of timesteps for which to train the model.
        seed (int | None): An optional seed for environment randomization.
    """
    path = "./sb3_log/multi/"
    logger = configure(path, ["stdout", "csv"])
    
    env = env_fn.parallel_env()
    env.reset(seed=seed)
    
    print(f"Starting training on {env}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    model_class = A2C if model_type.lower() == 'a2c' else PPO
    model = model_class("CnnPolicy", env, verbose=1, learning_rate=1e-3, ent_coef=0.01, vf_coef=0.5, stats_window_size=10)
    model.set_logger(logger)
    model.learn(total_timesteps=steps)
    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def eval_atari_supersuit(model_type, env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    """
    Evaluates a trained model (A2C or PPO) on the Boxing environment.

    Args:
        model_type (str): The type of RL model to use.
        num_games (int): The number of games to play for evaluation.
        render_mode (str | None): The rendering mode. Use 'human' for visual output or None for no rendering.
    """
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(glob.glob(f"./models/{env.metadata['name']}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    model_class = A2C if model_type.lower() == 'a2c' else PPO
    model = model_class.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}
    for i in range(num_games):
        env.reset(seed=i)
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()
        
    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print(f"Avg reward per agent, per game: ", avg_reward_per_agent)
    print(f"Full rewards: {rewards}")
    return avg_reward



def main():
    """
    Main function to train and evaluate RL models on Boxing Atari environment,
    and visualize performance of a trained model.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate multi-agent Atari environment models")
    parser.add_argument('--model', type=str, choices=['a2c', 'ppo'], default='a2c', help='Model type to use for training (A2C or PPO)')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train the model')
    parser.add_argument('--num_games', type=int, default=100, help='Number of games to evaluate the model')
    parser.add_argument('--render', action='store_true', help='Render the game visually during evaluation')
    parser.add_argument('--render_games', type=int, default=10, help='Number of games to render')

    args = parser.parse_args()

    env_fn = boxing_v2
    env_kwargs = {}
    
    print(f"Training the {args.model} model...")
    train_atari_supersuit(args.model, env_fn, steps=args.steps, seed=0, **env_kwargs)
    
    if args.render:
        print(f"Evaluating and rendering {args.render_games} games...")
        eval_atari_supersuit(args.model, env_fn, num_games=args.render_games, render_mode="human", **env_kwargs)
    
    print(f"Evaluating without rendering for {args.num_games} games...")
    eval_atari_supersuit(args.model, env_fn, num_games=args.num_games, render_mode=None, **env_kwargs)

if __name__ == "__main__":
    main()