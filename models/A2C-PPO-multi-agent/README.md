# Multi-Agent Reinforcement Learning for Atari Boxing Environment

## Introduction
This README details the setup and usage for training and evaluating reinforcement learning models (A2C or PPO) on the "boxing_v2" environment from PettingZoo's Atari games collection, using the SuperSuit library for preprocessing steps and the Stable Baselines3 framework.

## Features
- Training and evaluation with both A2C and PPO algorithms
- SuperSuit library for preprocessing steps such as frame stacking, color reduction, and resizing 
- Evaluation in both non-visual and visual modes to analyze agent behavior

## Code Explanation
### Configuration and Setup
The script is configured to work with PettingZoo environments.
The multi-agent environment is wrapped and modified with SuperSuit to make it compatible with the Stable Baselines3 library. This includes converting it into a vectorized environment suitable for batch training.
```python
env = boxing_v2.parallel_env()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_cpus=1, num_vec_envs=8, base_class='stable_baselines3')
```

### Training Process
The training process involves initializing the environment, setting up the necessary preprocessing wrappers, and configuring the A2C model with specific hyperparameters suited for multi-agent interaction. The trained model is saved with a timestamp.
```python
model = model_class("CnnPolicy", env, verbose=1, learning_rate=1e-3, ent_coef=0.01, vf_coef=0.5, stats_window_size=10)
model.learn(total_timesteps=steps)
model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

```

### Evaluation
Models are evaluated by playing a predefined number of games, and performance metrics such as average reward and rewards per agent are computed.
```python
latest_policy = max(glob.glob(f"./models/{env.metadata['name']}*.zip"), key=os.path.getctime)
model = model_class.load(latest_policy)
    # Run evaluation loop
```

### Visualization
For visualization, the rendering mode is set to 'human' to display the game window, allowing real-time observation of agent interactions and decisions.
```python
eval_atari_supersuit(boxing_v2, num_games=1, render_mode='human')
```

## Usage

The script can be run directly from the command line with various options to specify the model type, number of training steps, number of games for evaluation, and whether to render the games visually.
- **Basic usage**
```bash
python3 multi_agents.py --model a2c --steps 10000 --num_games 100
```

- **Training with high number of steps and evaluate without visual rendering**
```bash
python3 multi_agents.py --model ppo --steps 500000 --num_games 50
```

- **Evaluating with visual rendering**
```bash
python3 multi_agents.py --render --render_games 3
```