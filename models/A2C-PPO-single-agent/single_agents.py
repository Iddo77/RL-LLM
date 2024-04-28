import os
import argparse
import json
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

import torch
import optuna
        
    
def check_gpu():
    """
    Checks and prints whether PyTorch has access to a GPU.
    """
    if torch.cuda.is_available():
        print("GPU is available for training")
    else:
        print("Training on CPU; no GPU detected")

def train_and_evaluate(model_type, env_id, total_timesteps=int(1e6)):
    """
    Trains and evaluates a RL model on a specified Atari environment.

    Args:
        model_type (str): The type of RL model to use.
        env_id (str): The Gym ID of the Atari environment.
        total_timesteps (int): The total number of timesteps to train the model.

    Returns:
        mean_reward (float): The mean reward over the evaluation episodes.
        std_reward (float): The standard deviation of the reward over the evaluation episodes.
    """
    path = "./sb3_log/single/"
    logger = configure(path, ["stdout", "csv"])

    print(f"Training on: {env_id}")
    vec_env = make_atari_env(env_id, n_envs=16, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    model_class = A2C if model_type.lower() == 'a2c' else PPO
    with open("./models/hyperparameters.json", "r") as f:
        data = json.load(f)
        hyperparams = data[model_type][env_id]
    model = model_class("CnnPolicy", 
                        vec_env, 
                        verbose=1, 
                        stats_window_size=10,
                        **hyperparams)
                        # learning_rate=0.0003452643742071318,
                        # n_steps=342,
                        # gamma=0.9354550499669403)
    
    model.set_logger(logger)

    model.learn(total_timesteps)
    model.save(f"./models/{model_type.lower()}_{env_id}")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Evaluation on {env_id}: Mean reward: {mean_reward} +/- {std_reward}\n")
    return mean_reward, std_reward


def load_and_evaluate(model_type, env_id):
    """
    Loads and evaluates a saved RL model on a specified Atari environment.

    Args:
        model_type (str): The type of RL model to use.
        env_id (str): The Gym ID of the Atari environment.

    Returns:
        mean_reward (float): The mean reward over the evaluation episodes.
        std_reward (float): The standard deviation of the reward over the evaluation episodes.
    """
    vec_env = make_atari_env(env_id, n_envs=4, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model_class = A2C if model_type.lower() == 'a2c' else PPO
    model_path = f"./models/{model_class}_{env_id}.zip"
    model = model_class.load(model_path, env=vec_env)

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Evaluation on {env_id}: Mean reward: {mean_reward} +/- {std_reward}\n")
    return mean_reward, std_reward


def visualize_performance(model_type, env_id):
    """
    Renders and visually demonstrates the performance of a trained RL model on specified Atari environment.

    Args:
        model_type (str): The type of RL model to use.
        env_id (str): The Gym ID of the Atari environment
    """
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model_class = A2C if model_type.lower() == 'a2c' else PPO
    model = model_class.load(f"./models/{model_class}_{env_id}")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render("human")


def objective(model_type, trial, env_id):
    """
    Objective function for hyperparameter tuning using Optuna.

    Args:
        model_type (str): The type of RL model to use.
        trial (optuna.trial.Trial): Individual trial of the optimization process.
        env_id (str): The Gym ID of the Atari environment.        

    Returns:
        mean_reward (float): The mean reward achieved by the trained model on the evaluation.
    """
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int('n_steps', 128, 2048, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    ent_coef = trial.suggest_float('ent_coef', 0.0001, 0.1, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 0.5)

    vec_env = make_atari_env(env_id, n_envs=16, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model_class = A2C if model_type.lower() == 'a2c' else PPO
    model = model_class("CnnPolicy", vec_env, verbose=1,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                stats_window_size=10)
    
    model.learn(total_timesteps=100000)
    model.save(f"./models/a2c_tuned_{env_id}")

    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    return mean_reward


def main():
    """
    Main function to train and evaluate RL models on specified Atari environments,
    and visualize performance of a trained model.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate RL models on Atari environments.")
    parser.add_argument('--model', type=str, choices=['a2c', 'ppo'], default='a2c', help='Model type to use for training (A2C or PPO)')
    parser.add_argument('--env', type=str, choices=['BreakoutDeterministic-v4', 'PongDeterministic-v4', 'Boxing-v4'], default='BreakoutDeterministic-v4', help='Atari environment ID')
    parser.add_argument('--visualize', action='store_true', help='Visualize the performance of a trained model')
    parser.add_argument('--tuning', type=str, default='no', help='Enable hyperparameter tuning: yes or no')
    
    args = parser.parse_args()

    check_gpu()

    if args.tuning.lower() == 'yes':
            print(f"Starting hyperparameter tuning for environment: {args.env}")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(args.model, trial, args.env), n_trials=10)
            print("Best trial:")
            trial = study.best_trial
            print(f"    Value: {trial.value}")
            print("     Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

    else:
        model_path = f"./models/{args.model}_{args.env}_tuning{args.tuning}.zip"
        if os.path.exists(model_path):
            print(f"Model for {args.env} found. Loading and evaluating.")
            mean_reward, std_reward = load_and_evaluate(args.env)
        else:
            print(f"No model found for {args.env}. Training and evaluating.")
            mean_reward, std_reward = train_and_evaluate(args.model, args.env)
        
        print(f"Trained {args.model.upper()} on {args.env}: Mean Reward: {mean_reward} +/- {std_reward}")
        
        if args.visualize:
            print("Visualizing performance for", args.env)
            visualize_performance(args.env)

    mean_reward, std_reward = train_and_evaluate(args.model, args.env)
    print(f"Trained {args.model.upper()} on {args.env}: Mean Reward: {mean_reward} +/- {std_reward}")

    if args.visualize:
        visualize_performance(args.env)


if __name__ == '__main__':
    main()

