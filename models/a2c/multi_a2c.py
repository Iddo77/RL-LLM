from __future__ import annotations
import glob
import os
import time
import argparse

import supersuit as ss
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure

from pettingzoo.atari import boxing_v2

def train_atari_supersuit(env_fn, steps: int = 10_000, seed: int | None = 0, **env_kwargs):
    """
    Trains a A2C model on the Boxing environment from PettingZoo's Atari collection,
    with preprocessing steps.

    Args:
        steps (int): The total number of timesteps for which to train the model.
        seed (int | None): An optional seed for environment randomization.
    """ 
    path = "./sb3_log/multi/"
    logger = configure(path, ["stdout", "csv"])
    
    env = env_fn.parallel_env()

    env.reset(seed=seed)

    # SuperSuit wrappers
    # env = ss.color_reduction_v0(env, mode='B')  # Color reduction
    # env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize for CNN
    # env = ss.frame_stack_v1(env, 4)  # Frame stacking

    print(f"Starting training on {env}.")

    # Convert to VecEnv for Stable Baselines3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        ent_coef=0.01,
        vf_coef=0.5,
        stats_window_size=10,
    )
    model.set_logger(logger)

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

def eval_atari_supersuit(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    """
    Evaluates a trained A2C model on the Boxing environment from PettingZoo's Atari collection.
    
    Args:
        num_games (int): The number of games to play for evaluation.
        render_mode (str | None): The rendering mode. Use 'human' for visual output or None for no rendering.
    """
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Apply the same preprocessing as during training
    # env = ss.color_reduction_v0(env, mode='B')
    # env = ss.resize_v1(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 4)

    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(glob.glob(f"./models/{env.metadata['name']}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = A2C.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)
        
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
                #print(f"Action taken by {agent}: {act}")
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
    Main function to train and evaluate A2C models on Boxing Atari environment,
    and visualize performance of a trained model.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate A2C on multi-agent Atari environment")
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train the model')
    parser.add_argument('--num_games', type=int, default=100, help='Number of games to evaluate the model')
    parser.add_argument('--render', action='store_true', help='Render the game visually during evaluation')
    parser.add_argument('--render_games', type=int, default=10, help='Number of games to render')
    
    args = parser.parse_args()

    env_fn = boxing_v2
    env_kwargs = {}

    # Train the model
    print("Training the model...")
    train_atari_supersuit(env_fn, steps=args.steps, seed=0, **env_kwargs) 

    # Evaluate the model
    if args.render:
        print(f"Evaluating and rendering {args.render_games} games...")
        eval_atari_supersuit(env_fn, num_games=args.render_games, render_mode="human", **env_kwargs)
    
    print(f"Evaluating without rendering for {args.num_games} games...")
    eval_atari_supersuit(env_fn, num_games=args.num_games, render_mode=None, **env_kwargs)
    

if __name__ == "__main__":
    main()