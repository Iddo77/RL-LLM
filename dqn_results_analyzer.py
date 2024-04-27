import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DQN_BREAKOUT_500_FOLDER = r'.\models\DQN\2024-04-12_08.54_BreakoutDeterministic-v4'
DQN_PONG_500_FOLDER = r'.\models\DQN\2024-04-12_10.02_PongDeterministic-v4'
DQN_BOXING_500_FOLDER = r'.\models\DQN\2024-04-12_10.57_BoxingDeterministic-v4'
DQN_BREAKOUT_ALL_FOLDER = r'.\models\DQN\2024-04-13_21.56_BreakoutDeterministic-v4'
DQN_PONG_ALL_FOLDER = r'.\models\DQN\2024-04-13_17.30_PongDeterministic-v4'
DQN_BOXING_ALL_FOLDER = r'.\models\DQN\2024-04-13_13.28_BoxingDeterministic-v4'


def plot_average_score(csv_file, save_path, poly_degree=3):
    data = pd.read_csv(csv_file)

    x = data['end_episode']
    y = data['average_score']

    # Create a polynomial fit of specified degree
    coeffs = np.polyfit(x, y, poly_degree)
    # Generate a polynomial function from the coefficients
    poly_eq = np.poly1d(coeffs)
    # Create smooth x values for plotting the curve
    x_smooth = np.linspace(x.min(), x.max(), 500)
    # Generate y values from the polynomial function
    y_smooth = poly_eq(x_smooth)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', color='#1f77b4', label='Data Points')
    plt.plot(x_smooth, y_smooth, '-', color='#ff7f0e', label=f'{poly_degree}-Degree Polynomial Fit')
    plt.title('Average Score Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def print_averages(csv_file):

    df = pd.read_csv(csv_file)
    avg_score = df['average_score'].mean()
    print(f"Avg. Score: {avg_score:.1f}")

    total_time_steps = df['average_time_steps'].sum()
    first_5_percent_time_steps = total_time_steps * 0.05

    cumulative_time_steps = df['average_time_steps'].cumsum()
    first_5_percent_df = df[cumulative_time_steps <= first_5_percent_time_steps]
    first_5_percent_average_score = first_5_percent_df['average_score'].mean()
    print(f"First 5% Avg. Score: {first_5_percent_average_score:.1f}")

    last_5_percent_time_steps = total_time_steps * 0.95
    last_5_percent_df = df[cumulative_time_steps >= last_5_percent_time_steps]
    last_5_percent_average_score = last_5_percent_df['average_score'].mean()
    print(f"Final 5% Avg. Score: {last_5_percent_average_score:.1f}")


def print_all_averages():
    print("Breakout 500:")
    print_averages(os.path.join(DQN_BREAKOUT_500_FOLDER, 'episodes.csv'))
    print("Pong 500:")
    print_averages(os.path.join(DQN_PONG_500_FOLDER, 'episodes.csv'))
    print("Boxing 500:")
    print_averages(os.path.join(DQN_BOXING_500_FOLDER, 'episodes.csv'))
    print("Breakout:")
    print_averages(os.path.join(DQN_BREAKOUT_ALL_FOLDER, 'episodes.csv'))
    print("Pong:")
    print_averages(os.path.join(DQN_PONG_ALL_FOLDER, 'episodes.csv'))
    print("Boxing:")
    print_averages(os.path.join(DQN_BOXING_ALL_FOLDER, 'episodes.csv'))


def plot_all():

    os.makedirs(r'.\models\DQN-plots', exist_ok=True)

    filename_breakout_500 = os.path.join(DQN_BREAKOUT_500_FOLDER, 'episodes.csv')
    savepath_breakout_500 = r'.\models\DQN-plots\dqn-500-breakout-avg-scores.png'
    plot_average_score(filename_breakout_500, savepath_breakout_500)

    filename_pong_500 = os.path.join(DQN_PONG_500_FOLDER, 'episodes.csv')
    savepath_pong_500 = r'.\models\DQN-plots\dqn-500-pong-avg-scores.png'
    plot_average_score(filename_pong_500, savepath_pong_500)

    filename_boxing_500 = os.path.join(DQN_BOXING_500_FOLDER, 'episodes.csv')
    savepath_boxing_500 = r'.\models\DQN-plots\dqn-500-boxing-avg-scores.png'
    plot_average_score(filename_boxing_500, savepath_boxing_500)

    filename_breakout_all = os.path.join(DQN_BREAKOUT_ALL_FOLDER, 'episodes.csv')
    savepath_breakout_all = r'.\models\DQN-plots\dqn-all-breakout-avg-scores.png'
    plot_average_score(filename_breakout_all, savepath_breakout_all)

    filename_pong_all = os.path.join(DQN_PONG_ALL_FOLDER, 'episodes.csv')
    savepath_pong_all = r'.\models\DQN-plots\dqn-all-pong-avg-scores.png'
    plot_average_score(filename_pong_all, savepath_pong_all)

    filename_boxing_all = os.path.join(DQN_BOXING_ALL_FOLDER, 'episodes.csv')
    savepath_boxing_all = r'.\models\DQN-plots\dqn-all-boxing-avg-scores.png'
    plot_average_score(filename_boxing_all, savepath_boxing_all)


if __name__ == '__main__':
    print_all_averages()
    # plot_all()
