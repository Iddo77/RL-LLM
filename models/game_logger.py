import csv
import json
import os
import logging
from datetime import datetime

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

