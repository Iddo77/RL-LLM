# Atari Game Agents

This repository contains Python scripts for training Deep RL and LLM agents on Atari games.

## Prerequisites

- Python 3.11.7 or higher
- OpenAI Gym
- A `requirements.txt` file is included in the root directory, containing all necessary Python packages. Install them using `pip install -r requirements.txt`.

### Examples of Script Setups

1. **DQN Atari Agent (`dqn_atari.py`)**: Trains a Deep Q-Network agent.
2. **LLM Vision Agent (`llm_vision_agent.py`)**: Utilizes and LLM agent with GPT-4-vision to interpret the frames.
3. **LLM Agent OC Atari (`llm_agent_oc_atari.py`)**: Utilizes and LLM agent with OcAtari to interpret the frames.

## Running the Scripts

To run any of these scripts:

1. Ensure you have all the necessary dependencies installed.
2. Open a Python Integrated Development Environment (IDE).
3. Open the script you wish to run.
4. To play a different game, change the environment descriptor, such as `"PongDeterministic-v4"`,  in the `if __name__ == '__main__':` block to the desired game.
5. Execute the script in your IDE to start training the agent.

### Important Notes

- There is currently no extension for running these scripts directly from the console. You can run the scripts using an IDE or by calling them from other Python code.
- For specific details on the agents using Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C), refer to the respective READMEs in the `single_agent` and `multi_agent` folders.
