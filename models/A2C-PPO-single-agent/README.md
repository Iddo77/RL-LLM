# Single-Agent RL Model Training for Atari Environments

## Introduction
This README details the setup and usage for training and evaluating reinforcement learning models (A2C or PPO) on various Atari environments. Environments include BreakoutDeterministic-v4, PongDeterministic-v4, and Boxing-v4. 
The script utilizes the Stable Baselines3 framework for efficient RL training.

## Features
- Training and evaluation with A2C and PPO algorithms.
- Support for checking GPU availability to leverage CUDA for training if available.
- Evaluation of model performance with both visual and non-visual feedback.
- Optional hyperparameter tuning using Optuna to optimize model parameters.

## Code Explanation

### Configuration and Setup
- **Environment Setup**: Configures Atari environment and stacks frames for better state representation.
```python
env = make_atari_env('Breakout-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
```

### Training Process
- **Model initialization and saving**: Initalizes a model with specified hyperparameters, performs training, and saves trained model.
```python
model = A2C("CnnPolicy", env, verbose=1, **hyperparameters)
model.learn(total_timesteps=1e6)
model.save("./models/a2c_Breakout-v4")
```
- **Evaluation**: Loads a trained model and evaluates its performance.
```python
model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1e6)
model.save("./models/a2c_Breakout-v4")
```
- **Visualization**: Visualizes the model playing the game.
```python
visualize_performance('ppo', 'BreakoutDeterministic-v4')
```

## Usage

The script supports various command-line options to specify the model type, environment, and whether to perform hyperparameter tuning or visualize the model's performance.

- **Basic usage**
```bash
python3 single_agents.py --model ppo --env PongDeterministic-v4
```

- **Training with hyperparameter tuning and visualization**
```bash
python3 single_agents.py --model a2c --env Boxing-v4 --tuning yes --visualize
```

- **Evaluate and visualize pre-trained model**
```bash
python3 single_agents.py --model ppo --env BreakoutDeterministic-v4 --visualize
```