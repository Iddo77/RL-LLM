import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_data(log_dir, game):
    """
    Plots the training progress from the Stable Baselines3 logs.

    Args:
        log_dir (str): Directory where the CSV log files are stored.
        game (str): The game to plot the data for ('breakout' or 'pong').
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = ['500', 'None']
    models = ['a2c', 'ppo']
    linestyles = {'a2c': '-', 'ppo': '--'}

    for model in models:
        for condition in conditions:
            filename = f"{model}_{game}{condition}_progress.csv"
            file_path = os.path.join(log_dir, filename)
            data = pd.read_csv(file_path)

            if condition == 'None':
                label_condition = 'No max episodes'
            else:
                label_condition = f'{condition} max episodes'
            label = f'{model.upper()} - {label_condition}'
            ax.step(data['time/total_timesteps'], data['rollout/ep_rew_mean'], 
                    label=label, linestyle=linestyles[model], where='post')
            
    ax.set_xlabel('Total Timesteps')
    ax.set_ylabel('Average Reward per Episode')
    ax.set_title(f'{game.capitalize()} Score Progression')
    ax.legend()
    ax.grid(True)

    figure_path = os.path.join(log_dir, f"{game}_results.png")
    print(figure_path)
    plt.savefig(figure_path)
    plt.close()


def calculate_avg_scores(log_dir, model, game, episode_condition):
    """
    Calculates the average score for the first and last 5% timesteps for a given model and game.

    Args:
        log_dir (str): The directory where the log files are located.
        model (str): The model type ('a2c' or 'ppo').
        game (str): The game type ('breakout' or 'pong').
        episode_condition (str): The episode condition ('None' for uncapped or '500' for capped).
    
    Returns:
        Tuple of (first_5_avg, last_5_avg): The average scores for the first and last 5% of timesteps.
    """
    file_path = f"{log_dir}{model}_{game}{episode_condition}_progress.csv"

    data = pd.read_csv(file_path)

    total_timesteps = data['time/total_timesteps'].iloc[-1]

    threshold_first_5 = total_timesteps * 0.05
    threshold_last_5 = total_timesteps * 0.95

    data_first_5 = data[data['time/total_timesteps'] <= threshold_first_5]
    data_last_5 = data[data['time/total_timesteps'] >= threshold_last_5]

    first_5_avg = data_first_5['rollout/ep_rew_mean'].mean()
    last_5_avg = data_last_5['rollout/ep_rew_mean'].mean()
    
    return first_5_avg, last_5_avg


def main():
    """
    Main function to calculate average score for first and last 5% of time steps and plotting results.
    """
    log_dir = './sb3_log/single/'

    results = {}
    for model in ['a2c', 'ppo']:
        for game in ['breakout', 'pong']:
            for condition in ['None', '500']:
                results[(model, game, condition)] = calculate_avg_scores(log_dir, model, game, condition)

    print(results)

    # Plotting
    plot_training_data(log_dir, 'breakout')
    plot_training_data(log_dir, 'pong')

if __name__ == '__main__':
    main()