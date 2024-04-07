import csv
import json
import os
import logging
from datetime import datetime

import numpy as np

from models.game_info import GameInfo


class GameLogger:
    def __init__(self, model_name: str, game_info: GameInfo):
        self.model_name = model_name
        self.game_info = game_info
        base_folder = os.path.join(os.path.dirname(__file__), model_name)
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H.%M')}_{game_info.name.capitalize()}"
        self.log_folder = os.path.join(base_folder, filename)
        os.makedirs(self.log_folder, exist_ok=True)
        self.init_logging()
        self.game_log = os.path.join(self.log_folder, 'game_log.jsonl')
        self.llm_log = os.path.join(self.log_folder, 'llm_results_log.jsonl')
        csv_file_path = os.path.join(self.log_folder, 'rewards.csv')
        self.csv_file = open(csv_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['episode', 'time_step', 'action', 'lives', 'reward', 'score'])

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"Error closing resources in GameLogger: {e}")

    def init_logging(self):
        log_filename = os.path.join(self.log_folder, 'error.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("LLM Agent logging initialized.")

    def log_game_data(self, i_episode, time_step, action, lives, reward, score, game_state):
        self.csv_writer.writerow([i_episode, time_step, action, lives, reward, score])
        print(f'Episode: {i_episode}  Time step: {time_step}  Action: {self.game_info.actions[action]}  '
              f'Lives:  {lives}  Reward: {reward}  Score: {score}')
        self.log_game_event(i_episode, time_step, lives, action, reward, score, game_state)

    def log_game_event(self, episode, time_step, lives, action, reward, score, game_state):
        event = {
            "episode": episode,
            "time_step": time_step,
            "lives": lives,
            "action": action,
            "reward": reward,
            "score": score,
            "game_state": game_state.to_json()
        }
        with open(self.game_log, 'a') as file:
            json.dump(event, file)
            file.write('\n')

    def log_llm_result(self, episode, time_step, llm_result):
        interaction = {
            "episode": episode,
            "time_step": time_step,
            "llm_result": llm_result
        }
        with open(self.llm_log, 'a') as file:
            json.dump(interaction, file)
            file.write('\n')

    def close(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None


class DQNGameLogger:
    def __init__(self, game_name, log_episode_interval=10, log_statistics_interval=100):
        self.game_name = game_name
        self.log_episode_interval = log_episode_interval
        self.save_statistics_interval = log_statistics_interval
        base_folder = os.path.join(os.path.dirname(__file__), 'DQN')
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H.%M')}_{game_name}"
        self.log_folder = os.path.join(base_folder, filename)
        os.makedirs(self.log_folder, exist_ok=True)

        episode_log_file_path = os.path.join(self.log_folder, 'episodes.csv')
        self.episode_log_file = open(episode_log_file_path, mode='w', newline='')
        self.episode_log_writer = csv.writer(self.episode_log_file)
        self.episode_log_writer.writerow(['start_episode', 'end_episode', 'average_time_steps', 'average_score',
                                          'high_score', 'start_eps', 'end_eps'])

        self.save_weights_statistics_file_path = os.path.join(self.log_folder, 'save_weights_statistics.csv')
        self.save_weights_statistics_file = open(episode_log_file_path, mode='w', newline='')
        self.save_weights_statistics_writer = csv.writer(self.episode_log_file)
        self.save_weights_statistics_writer.writerow(['start_episode', 'end_episode', 'average_time_steps', 'average_score',
                                                  'high_score', 'start_eps', 'end_eps'])

        self.last_logged_episode = 0
        self.last_logged_statistics_episode = 0
        self.scores = []
        self.time_steps = []
        self.eps_history = []

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"Error closing resources in GameLogger: {e}")

    def log_episode(self, episode, score, time_steps, eps):

        self.scores.append(score)
        self.time_steps.append(time_steps)
        self.eps_history.append(eps)

        if episode % self.log_episode_interval == 0:
            start_episode = self.last_logged_episode + 1
            average_time_steps = int(np.mean(self.time_steps[-self.log_episode_interval:]))
            interval_scores = self.scores[-self.log_episode_interval:]
            average_score = round(np.mean(interval_scores), 2)
            high_score = max(interval_scores)
            eps_values = self.eps_history[-self.log_episode_interval:]
            start_eps = round(eps_values[0], 4)
            end_eps = round(eps, 4)
            self.episode_log_writer.writerow([start_episode, episode, average_time_steps, average_score,
                                              high_score, start_eps, end_eps])
            print(f"Episode: {start_episode} to {episode}  Average time steps: {average_time_steps}  "
                  f"Average score: {average_score}  High score: {high_score}  Epsilon: {start_eps} to {end_eps}")
            self.last_logged_episode = episode

        if episode % self.save_statistics_interval == 0:
            self.log_save_weights(episode)

    def log_save_weights(self, episode):
        start_episode = self.last_logged_statistics_episode + 1
        average_time_steps = int(np.mean(self.time_steps[-self.save_statistics_interval:]))
        interval_scores = self.scores[-self.save_statistics_interval:]
        average_score = round(np.mean(interval_scores), 2)
        high_score = max(interval_scores)
        eps_values = self.eps_history[-self.save_statistics_interval:]
        start_eps = round(eps_values[0], 4)
        end_eps = round(eps_values[-1], 4)
        self.save_weights_statistics_writer.writerow([start_episode, episode, average_time_steps, average_score,
                                                      high_score, start_eps, end_eps])
        print(f"Episode: {start_episode} to {episode}  Average time steps: {average_time_steps}  "
              f"Average score: {average_score}  High score: {high_score}  Epsilon: {start_eps} to {end_eps}")
        self.last_logged_statistics_episode = episode

    def close(self):
        if self.episode_log_file is not None:
            self.episode_log_file.close()
            self.episode_log_file = None
            self.episode_log_writer = None
        if self.save_weights_statistics_file is not None:
            self.save_weights_statistics_file.close()
            self.save_weights_statistics_file = None
            self.save_weights_statistics_writer = None
