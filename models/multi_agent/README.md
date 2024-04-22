# Multi-Agent A2C Training for Atari Boxing Environment

## Introduction
This section extends our A2C training capabilities to a multi-agent context using the PettingZoo library, which is a collection of environments for multi-agent reinforcement learning. Specifically, we demonstrate training with the "boxing_v2" environment from PettingZoo's Atari games collection.

## Features
- Multi-agent training and evaluation with the A2C algorithm
- Usage of SuperSuit for preprocessing steps like frame stacking, color reduction, and resizing, tailored for Atari games
- Evaluation in both non-visual and visual modes to analyze agent behavior

## Code Explanation
### Configuration and Setup
The multi-agent environment is wrapped and modified with SuperSuit to make it compatible with the Stable Baselines3 library. This includes converting it into a vectorized environment suitable for batch training.
```python
env = boxing_v2.parallel_env()
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_cpus=1, num_vec_envs=8, base_class='stable_baselines3')
```

### Training Process
The training process involves initializing the environment, setting up the necessary preprocessing wrappers, and configuring the A2C model with specific hyperparameters suited for multi-agent interaction.
```python
model = A2C(
    "CnnPolicy",
    env,
    verbose=1, 
    learning_rate=1e-3,
    ent_coef=0.01,
    vf_coef=0.5,
    stats_window_size=10,
)
model.learn(total_timesteps=1000000)
model.save("path_to_save") 
```

### Evaluation
Evaluation is done by loading the trained model and running it through a set number of games, tracking the rewards for each agent. 
```python
def eval_atari_supersuit(env_fn, num_games=10, render_mode=None):
    env = env_fn.env(render_mode=render_mode)
    model = A2C.load("path_to_model")
    # Run evaluation loop
```

### Visualization
For visualization, the rendering mode is set to 'human' to display the game window, allowing real-time observation of agent interactions and decisions.
```python
eval_atari_supersuit(boxing_v2, num_games=1, render_mode='human')
```

## Run

The `multi_a2c.py` script is used for training and evaluation of the A2C model in a multi-agent setting using the Boxing environment from PettingZoo's Atari collection.

- **Basic training and evaluation**: this command trains and evaluates the model with default settings and without rendering.
```bash
python3 multi_a2c.py
```

- **Training with specified steps and games**: this command trains and evaluates the model for 500,000 steps, and 50 games.
```bash
python3 multi_a2c.py --steps 500000 --num_games 50
```

- **Evaluating with rendering for 2 games (assuming training is done)**: this command evaluates the model with rendering for 3 games.
```bash
python3 multi_a2c.py --render --render_games 3
```