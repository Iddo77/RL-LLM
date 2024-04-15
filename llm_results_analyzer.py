import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LLM_VISION_BREAKOUT_FOLDER = r'.\models\LLM-Vision-Agent\2024-04-07_12.09_Breakout'
LLM_VISION_PONG_FOLDER = r'.\models\LLM-Vision-Agent\2024-04-09_07.19_Pong'
LLM_VISION_BOXING_FOLDER = r'.\models\LLM-Vision-Agent\2024-04-08_15.35_Boxing'
LLM_OCATARI_BREAKOUT_FOLDER = r'.\models\LLM-Agent-OcAtari\2024-04-08_11.48_Breakout'
LLM_OCATARI_PONG_FOLDER = r'.\models\LLM-Agent-OcAtari\2024-04-08_13.58_Pong'
LLM_OCATARI_BOXING_FOLDER = r'.\models\LLM-Agent-OcAtari\2024-04-08_12.42_Boxing'


def plot_cumulative_rewards(df1, df2, save_path, game_name):
    df1['cumulative_reward'] = df1.groupby('episode')['reward'].cumsum()
    df2['cumulative_reward'] = df2.groupby('episode')['reward'].cumsum()

    max_episode = max(df1['episode'].max(), df2['episode'].max())
    episodes = range(1, max_episode + 1)

    rewards1 = [df1[df1['episode'] == ep]['cumulative_reward'].iloc[-1] if ep in df1['episode'].values else 0 for ep in episodes]
    rewards2 = [df2[df2['episode'] == ep]['cumulative_reward'].iloc[-1] if ep in df2['episode'].values else 0 for ep in episodes]

    plt.plot(episodes, rewards1, label='LLM-Vision')
    plt.plot(episodes, rewards2, label='LLM-OcAtari')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(f'{game_name} Cumulative Reward Comparison')
    plt.legend()
    plt.xticks(range(1, 6))
    plt.savefig(save_path)
    plt.close()


def prepare_data(df):
    # Sort by episode and time_step to ensure the order is correct
    # Create a new column 'cumulative_time_step' that accumulates the time steps across episodes
    df['cumulative_time_step'] = df.groupby('episode')['time_step'].cumsum()
    df['cumulative_time_step'] = df['cumulative_time_step'] + df['episode'].map(df.groupby('episode')['cumulative_time_step'].last().cumsum().shift(1, fill_value=0))
    return df


def plot_cumulative_scores(df1, df2, save_path, game_name):
    # Use index + 1 as the time step
    time_steps_1 = df1.index + 1
    time_steps_2 = df2.index + 1

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps_1, df1['score'], label='Vision LLM', marker='o')
    plt.plot(time_steps_2, df2['score'], label='Ocatari LLM', marker='x')

    plt.xlabel('Cumulative Time Step')
    plt.ylabel('Score')
    plt.title(f'{game_name} Score Comparison Per Cumulative Time Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def get_interpolated_scores_per_episode(df, num_steps=100):
    episodes = df['episode'].unique()
    interpolated_scores = np.zeros((len(episodes), num_steps))

    for i, episode in enumerate(episodes):
        sub_df = df[df['episode'] == episode]
        if len(sub_df) > 1:
            interp_func = np.interp(
                np.linspace(1, sub_df['time_step'].max(), num=num_steps),  # Target new timesteps
                sub_df['time_step'],
                sub_df['score']
            )
        else:
            # If only one data point, repeat the score
            interp_func = np.repeat(sub_df['score'].iloc[0], num_steps)
        interpolated_scores[i] = interp_func

    return episodes, interpolated_scores


def plot_interpolated_scores_per_episode(df1, df2, save_path, game_name):
    episodes1, scores1 = get_interpolated_scores_per_episode(df1)
    episodes2, scores2 = get_interpolated_scores_per_episode(df2)
    max_episode = max(episodes1.max(), episodes2.max())

    # Plot all episodes for Vision LLM
    for i, ep in enumerate(episodes1):
        if i == 0:
            plt.plot(np.linspace(ep, ep + 0.99, 100), scores1[i], label='LLM-Vision', color='#1f77b4')
        else:
            plt.plot(np.linspace(ep, ep + 0.99, 100), scores1[i], color='#1f77b4')

    # Plot all episodes for OcAtari LLM
    for i, ep in enumerate(episodes2):
        if i == 0:
            plt.plot(np.linspace(ep, ep + 0.99, 100), scores2[i], label='LLM-OcAtari', color='#ff7f0e',
                     linestyle='-.')
        else:
            plt.plot(np.linspace(ep, ep + 0.99, 100), scores2[i], color='#ff7f0e', linestyle='-.')

    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'{game_name} Score Progression per Episode')

    def episode_formatter(x, pos):
        if x > max_episode:  # Hide episodes labels not in dataset
            return ''
        else:
            return int(x)

    plt.gca().xaxis.set_major_formatter(episode_formatter)

    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_comparisons(df_vision_llm_rewards, df_ocatari_llm_rewards, game_name):
    plot_cumulative_rewards(df_vision_llm_rewards, df_ocatari_llm_rewards,
                            fr'.\models\LLM-comparison\{game_name}-cumulative-reward.png', game_name)
    prepare_data(df_vision_llm_rewards)
    prepare_data(df_ocatari_llm_rewards)
    plot_cumulative_scores(df_vision_llm_rewards, df_ocatari_llm_rewards,
                           fr'.\models\LLM-comparison\{game_name}-scores.png', game_name)
    plot_interpolated_scores_per_episode(df_vision_llm_rewards, df_ocatari_llm_rewards,
                                         fr'.\models\LLM-comparison\{game_name}-interpolated-scores-per-episide.png',
                                         game_name)


def compare_all_games():
    # Breakout
    df_vision_llm_rewards = pd.read_csv(os.path.join(LLM_VISION_BREAKOUT_FOLDER, 'rewards.csv'))
    df_ocatari_llm_rewards = pd.read_csv(os.path.join(LLM_OCATARI_BREAKOUT_FOLDER, 'rewards.csv'))
    plot_comparisons(df_vision_llm_rewards, df_ocatari_llm_rewards, 'Breakout')
    # Pong
    df_vision_llm_rewards = pd.read_csv(os.path.join(LLM_VISION_PONG_FOLDER, 'rewards.csv'))
    df_ocatari_llm_rewards = pd.read_csv(os.path.join(LLM_OCATARI_PONG_FOLDER, 'rewards.csv'))
    plot_comparisons(df_vision_llm_rewards, df_ocatari_llm_rewards, 'Pong')
    # Boxing
    df_vision_llm_rewards = pd.read_csv(os.path.join(LLM_VISION_BOXING_FOLDER, 'rewards.csv'))
    df_ocatari_llm_rewards = pd.read_csv(os.path.join(LLM_OCATARI_BOXING_FOLDER, 'rewards.csv'))
    plot_comparisons(df_vision_llm_rewards, df_ocatari_llm_rewards, 'Boxing')


if __name__ == '__main__':
    compare_all_games()
