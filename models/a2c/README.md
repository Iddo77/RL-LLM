# A2C Single Agent Model Training for Atari Environments

## Introduction
This project uses the synchronous, deterministic variant of Asynchronous Advantage Actor Critic algorithm (A2C) implemented via Stable Baselines3 to train models on various Atari game environments. 

## Features
- Training A2C models on Atari environments
- Evaluating model performance and displaying average rewards
- Visualizing the trained model in action within the Atari game

## Code Explanation

### Configuration and Setup
- **Logging Setup**: We configure a logger to  save output to both the console and a CSV file.
```python
log_dir = "./sb3_log/"
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["stdout", "csv"])
```

### Callbacks
- **MaxEpisodesCallback**: This callback is designed to stop training once a maximum number of episodes is reached.
```python
class MaxEpisodesCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        if 'episode' in self.locals:
            self.episode_count += 1
        return self.epsiode_count < self.max_episodes
```

### Core Functions
- **train_and_evaluate**: Sets up the environment, initializes A2C model, applies callbacks, and performs training. After training, it evaluates the model's performance.
```python
def train_and_evaluate(env_id, total_timesteps=int(1e6), max_episodes=None):
    vec_env = make_atari_env(env_id, n_envs=16, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = A2C("CnnPolicy",
                vec_env,
                verbose=1,
                # learning_rate=2.5e-4,
                # gamma=0.99,
                # n_steps=5,
                ent_coef=0.01,
                vf_coef=0.25,
                # normalize_advantage=True
                stats_window_size=10 # number of episodes to average success, episode, reward
                )
    model.set_logger(logger)
    ...
```
- **load_and_evaluate**: Loads a trained model and evaluates its performance.
```python
def load_and_evaluate(env_id):
    vec_env = make_atari_env(env_id, n_envs=4, seed=0)
    model_path = f"a2c_{env_id}.zip"
    model = A2C.load(model_path, env=vec_env)
    ...
```
- **visualize_performance**: Visualizes the model playing the game.
```python
def visualize_performance(env_id):
    env = make_atari_env(env_id, n_envs=1, seed=0)
    model = A2C.load(f"a2c_{env_id}")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.steps(action)
        env.render("human")
```
- **main**: Runs training and evaluation, handling GPU checks, summarizing results.
```python
def main():
    check_gpu()
    environments = ["BreakoutDeterministic-v4", "PongDeterministic-v4", "Boxing-v4"]
    ...
```

## Run

The `a2c.py` script is designed to train and evaluate the A2C model on single Atari game environments. It supports hyperparameter tuning, selecting specific Atari environments, setting the number of episodes, and visualizing the trained model.

- **Basic training and evaluation**: this command trains and evaluates the A2C model on the BreakoutDeterministic-v4 environment with default settings.
```bash
python3 a2c.py --env BreakoutDeterministic-v4
```

- **Training with hyperparameter tuning**: this command enables hyperparameter tuning for the PongDeterministic-v4 environment.
```bash
python3 a2c.py --env PongDeterministic-v4 --tuning yes
```

- **Setting maximum episodes**: this command sets a cap of 500 episodes for training the model on the BreakoutDeterministic-v4 environment.
```bash
python3 a2c.py --env BreakoutDeterministic-v4 --max_episodes 500
```

- **Visualizing the model**: this command trains the model and then visualizes its performance in the game.
```bash
python3 a2c.py --env PongDeterministic-v4 --visualize
```

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