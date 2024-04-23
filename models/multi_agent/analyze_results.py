import os
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def plot_results(data, title):
    """
    Plots the training results for a given model based on timesteps and rewards.
    
    Args:
        data (DataFrame): Data containing the fields 'timesteps' and 'reward'.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data['timesteps'], data['reward'], label='Reward')
    plt.title(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_data(data, model_name):
    """
    Summarizes the data by calculating average rewards and other statistics.
    
    Args:
        data (DataFrame): Data containing the rewards.

    Returns:
        dict: A dictionary containing summary statistics like average and max rewards.
    """
    summary = {
        'Average Reward': data['reward'].mean(),
        'Max Reward': data['reward'].max(),
        'Min Reward': data['reward'].min(),
        'Total Timesteps': data['timesteps'].iloc[-1]
    }
    return summary

def main():
    """
    Main function to load data, create plots, and display summary statistics.
    """

    a2c_data = pd.read_csv(os.path.join('./models/multi_agent/sb3_log/multi/a2c_multi_progress.csv'))
    ppo_data = pd.read_csv(os.path.join('./models/multi_agent/sb3_log/multi/ppo_multi_progress.csv'))
    print(a2c_data)
    plot_results(a2c_data, 'A2C Training Progress')
    plot_results(ppo_data, 'PPO Training Progress')
    
    a2c_summary = summarize_data(a2c_data)
    ppo_summary = summarize_data(ppo_data)

    print(tabulate([a2c_summary, ppo_summary], headers="keys", tablefmt='grid'))


if __name__ == '__main__':
    main()