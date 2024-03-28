import torch
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

def train_and_evaluate(env_id, total_timesteps=int(10_000)):
    """
    Trains and evaluates an A2C model on a specified Atari environment.

    Args:
        env_id (str): The Gym ID of the Atari environment.
        total_timesteps (int): The total number of timesteps to train the model.

    Returns:
        mean_reward (float): The mean reward over the evaluation episodes.
        std_reward (float): The standard deviation of the reward over the evaluation episodes.
    """
    print(f"Training on: {env_id}")
    vec_env = make_atari_env(env_id, n_envs=4, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = A2C("CnnPolicy",
                vec_env,
                verbose=1,
                learning_rate=2.5e-4,
                gamma=0.99,
                n_steps=5,
                ent_coef=0.01,
                vf_coef=0.5,
                normalize_advantage=True)
    
    model.learn(total_timesteps=total_timesteps)
    model.save(f"a2c_{env_id}")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
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

    model = A2C.load(f"a2c_{env_id}")

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

def main():
    """
    Main function to train and evaluate A2C models on specified Atari environments,
    and visualize performance of a trained model.
    """
    check_gpu()

    environments = ["BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "BoxingNoFrameskip-v4"]
    evaluation_results = {}

    for env in environments:
        mean_reward, std_reward = train_and_evaluate(env)
        evaluation_results[env] = (mean_reward, std_reward)

    print("Evaluation Summary:")
    for env, results in evaluation_results.items():
        print(f"{env}: Mean Reward: {results[0]} +/- {results[1]}")
    
    # Optional: visualize performance of one of the environments
    visualize_performance("BreakoutNoFrameskip-v4")


if __name__ ==  '__main__':
    main()