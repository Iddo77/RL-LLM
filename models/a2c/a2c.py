import os
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch
import optuna
import argparse

class MaxEpisodesCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            self.episode_count += 1
        return self.episode_count < self.max_episodes
    

def train_and_evaluate(env_id, total_timesteps=int(1e6), max_episodes=500):
    """
    Trains and evaluates an A2C model on a specified Atari environment.

    Args:
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

    model = A2C("CnnPolicy", 
                vec_env, 
                verbose=1,
                stats_window_size=10 # number of episodes to average success, episode, reward
                )
    model.set_logger(logger)

    if max_episodes is not None:
        callback = MaxEpisodesCallback(max_episodes=max_episodes)
        model.learn(total_timesteps, callback=callback)
    else:
        model.learn(total_timesteps=total_timesteps)
    model.save(f"./models/a2c_{env_id}")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Evaluation on {env_id}: Mean reward: {mean_reward} +/- {std_reward}\n")

    return mean_reward, std_reward

def load_and_evaluate(env_id):
    """
    Loads and evaluates a saved A2C model on a specified Atari environment.

    Args:
        env_id (str): The Gym ID of the Atari environment.

    Returns:
        mean_reward (float): The mean reward over the evaluation episodes.
        std_reward (float): The standard deviation of the reward over the evaluation episodes.
    """
    vec_env = make_atari_env(env_id, n_envs=4, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model_path = f"./models/a2c_{env_id}.zip"
    model = A2C.load(model_path, env=vec_env)

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Evaluation on {env_id}: Mean reward: {mean_reward} +/- {std_reward}\n")
    return mean_reward, std_reward


def visualize_performance(env_id):
    """
    Renders and visually demonstrates the performance of a trained A2C model on specified Atari environment.

    Args:
        env_id (str): The Gym ID of the Atari environment
    """
    env = make_atari_env(env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = A2C.load(f"./models/a2c_{env_id}")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render("human")


def check_gpu():
    """
    Checks and prints whether PyTorch has access to a GPU.
    """
    if torch.cuda.is_available():
        print("GPU is available for training")
    else:
        print("Training on CPU; no GPU detected")


def objective(trial, env_id):
    """
    Objective function for hyperparameter tuning using Optuna.

    Args:
        env_id (str): The Gym ID of the Atari environment.
        trial (optuna.trial.Trial): Individual trial of the optimization process.

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

    model = A2C("CnnPolicy", vec_env, verbose=1,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                stats_window_size=10)
    #model.set_logger(logger)

    callback = MaxEpisodesCallback(max_episodes=500)
    model.learn(total_timesteps=100000, callback=callback)
    model.save(f"./models/a2c_tuned_{env_id}")

    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    return mean_reward


def main():
    """
    Main function to train and evaluate A2C models on specified Atari environments,
    and visualize performance of a trained model.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a A2C model on Atari environments.")
    parser.add_argument('--tuning', type=str, default='no', help='Enable hyperparameter tuning: yes or no')
    parser.add_argument('--env', type=str, default='all', help='Atari environment ID or "all" for all environments')
    parser.add_argument('--visualize', action='store_true', help='Visualize the performance of a trained model')
    parser.add_argument('--max_episodes', type=str, default='500', help='Maximum number of episodes to train')

    args = parser.parse_args()
    max_episodes = None if args.max_episodes.lower() == 'none' else int(args.max_episodes)

    check_gpu()

    # Determine environments to process
    environments = [args.env] if args.env.lower() != 'all' else ["BreakoutDeterministic-v4", "PongDeterministic-v4"]#, "Boxing-v4"]
    
    # Configure logger with a dynamic filename based on max_episodes
    for env in environments:
        #logger = configure_logger(env, max_episodes, args.tuning)

        if args.tuning.lower() == 'yes':
            print(f"Starting hyperparameter tuning for environment: {env} with max episodes: {max_episodes}")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, env), n_trials=10)
            print("Best trial:")
            trial = study.best_trial
            print(f"    Value: {trial.value}")
            print("     Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        else:
            model_path = f"./models/a2c_{env}_maxep{max_episodes}_tuning{args.tuning}.zip"
            if os.path.exists(model_path):
                print(f"Model for {env} found. Loading and evaluating.")
                mean_reward, std_reward = load_and_evaluate(env)
            else:
                print(f"No model found for {env}. Training and evaluating.")
                mean_reward, std_reward = train_and_evaluate(env, max_episodes=max_episodes)
            #logger.close()
            print(f"{env}: Mean Reward: {mean_reward} +/- {std_reward}")
            
            if args.visualize:
                print("Visualizing performance for", env)
                visualize_performance(env)


if __name__ ==  '__main__':
    main()