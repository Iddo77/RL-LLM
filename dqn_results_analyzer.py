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


# def plot_average_score(csv_file, save_path):
#     data = pd.read_csv(csv_file)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(data['end_episode'], data['average_score'], marker='o')
#     plt.title('Average Score Over Episodes')
#     plt.xlabel('End Episode')
#     plt.ylabel('Average Score')
#     plt.grid(True)
#
#     plt.savefig(save_path)
#     plt.close()


# def plot_average_score(csv_file, save_path):
#     # Load the data from the CSV file
#     data = pd.read_csv(csv_file)
#
#     # Extracting x and y values for plotting
#     x = data['end_episode']
#     y = data['average_score']
#
#     # Create a polynomial fit of degree 2 (quadratic)
#     coeffs = np.polyfit(x, y, 2)
#     # Generate a polynomial function from the coefficients
#     poly_eq = np.poly1d(coeffs)
#     # Create smooth x values for plotting the curve
#     x_smooth = np.linspace(x.min(), x.max(), 500)
#     # Generate y values from the polynomial function
#     y_smooth = poly_eq(x_smooth)
#
#     # Plotting the data
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, 'bo', label='Data Points')  # 'bo' for blue circle markers
#     plt.plot(x_smooth, y_smooth, 'r-', label='Fit Curve')  # 'r-' for red line
#     plt.title('Average Score Over Episodes')
#     plt.xlabel('End Episode')
#     plt.ylabel('Average Score')
#     plt.grid(True)
#     plt.legend()
#
#     # Save the plot to the specified path
#     plt.savefig(save_path)
#     plt.close()

def plot_average_score(csv_file, save_path, poly_degree=3):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file)

    # Extracting x and y values for plotting
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

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'bo', label='Data Points')  # 'bo' for blue circle markers
    plt.plot(x_smooth, y_smooth, 'r-', label=f'{poly_degree}-Degree Polynomial Fit')  # 'r-' for red line
    plt.title('Average Score Over Episodes')
    plt.xlabel('End Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()

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
    plot_all()
