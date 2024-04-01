import json
import os
import logging
import csv
import numpy as np
import gymnasium as gym
from datetime import datetime
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

from models.game_info import GameInfo
from models.game_state import GameState
from image_processing import preprocess_frame, convert_image_to_base64, merge_images_with_bars, save_image_to_file
from utils import parse_json_from_substring, escape_brackets


# and environment variable OPENAI_API_KEY must be set with the OpenAI key


def query_image_with_text(image, text):

    base64_image = convert_image_to_base64(image)
    chat = ChatOpenAI(model='gpt-4-vision-preview', max_tokens=256)
    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ]
            )
        ]
    )

    return response


def get_llm_messages_to_update_game_state():
    prompt_text = """### GAME STATE
{game_state}

### INSTRUCTIONS
- Describe all entities in the current game state as detailed in recent_frames_and_motion_summary, including their position and motion. Reuse existing entities as much as possible.
- Describe a world model in a few short sentences.
- Give the next action, by choosing one from the available actions. Try to vary your actions if you don't know what to do.

### RESULT
Create a json containing the world model, the entities, and the next action. Write nothing else. For example:
{
    "world_model": "A game to hunt ducks"
    "entities": [{"name": "duck", "position": "bottom-left", "motion": "moving to the left"}],
    "next_action": "FIRE"
}"""

    # escape brackets in json, otherwise validation of langchain will fail
    prompt_text = escape_brackets(prompt_text, ['game_state'])

    system_message = SystemMessage("You are an RL agent playing an Atari game.")
    prompt_template = HumanMessagePromptTemplate.from_template(input_variables=["game_state"], template=prompt_text)
    return [system_message, prompt_template]


def get_llm_message_life_lost(is_game_over: bool):

    if is_game_over:
        prompt_start = 'Game over!!!'
    else:
        prompt_start = 'You lost a life!'

    prompt_text = prompt_start + """
    
Do you want to to update the guidelines for future games?

Respond with a json like this:

   {
       "guidelines": {
            "recommended_actions": [],
            "actions_to_avoid": []
        }
   }

When responding, make sure to copy all existing guidelines that you want to keep. You can expand the list if you think you can do better in the future. 
If you have no new ideas, then just copy the old guidelines without adding anything new. 

Respond with the json only and write nothing else.
    """
    # escape brackets in json, otherwise validation of langchain will fail
    prompt_text = escape_brackets(prompt_text, [])

    return HumanMessagePromptTemplate.from_template(template=prompt_text)


def get_llm_message_game_reward():
    prompt_text = """You received a reward following your last action! That means you did something right. 

    Do you want to to update the guidelines for future games?

    Respond with a json like this:

       {
           "guidelines": {
                "recommended_actions": [],
                "actions_to_avoid": []
            }
       }

    When responding, make sure to copy all existing guidelines that you want to keep. You can expand the list if you think you know why you got the reward.
    If you have no new ideas, then just copy the old guidelines without adding anything new. 

    Respond with the json only and write nothing else.
    """

    # escape brackets in json, otherwise validation of langchain will fail
    prompt_text = escape_brackets(prompt_text, [])

    return HumanMessagePromptTemplate.from_template(template=prompt_text)


def invoke_llm_and_parse_result(llm: ChatOpenAI,
                                llm_messages: list[SystemMessage | HumanMessagePromptTemplate | AIMessage],
                                game_state: GameState):
    chat_prompt_template = ChatPromptTemplate.from_messages(llm_messages)
    chain = LLMChain(llm=llm, prompt=chat_prompt_template)
    result = chain.invoke({"game_state": game_state.to_json()})
    return parse_json_from_substring(result["text"])


def update_game_state_and_act(llm_result: dict, game_state: GameState, game_info: GameInfo):

    if "entities" in llm_result:
        valid_entities = []
        entities = llm_result["entities"]
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    name = ent.get("name")
                    position = ent.get("position")
                    if name is not None and position is not None:
                        if "motion" not in ent:
                            ent["motion"] = "unknown"
                        valid_entities.append(ent)
                        if name not in game_state.previously_encountered_entities:
                            game_state.previously_encountered_entities.append(name)

        if valid_entities:
            game_state.entities_in_previous_game_state = valid_entities

    if "world_model" in llm_result:
        game_state.world_model = llm_result["world_model"]

    action = 0  # NOOP
    if "next_action" in llm_result:
        action_text = llm_result["next_action"]
        if action_text in game_info.actions:
            action = game_info.actions.index(action_text)

    return action


def update_guidelines(llm_result: dict, game_state: GameState):
    if "guidelines" in llm_result:
        new_guidelines = llm_result["guidelines"]
        if "recommended_actions" in new_guidelines:
            new_recommended_actions = new_guidelines["recommended_actions"]
            if len(new_recommended_actions) >= len(game_state.guidelines["recommended_actions"]):
                game_state.guidelines["recommended_actions"] = new_recommended_actions
        if "actions_to_avoid" in new_guidelines:
            new_actions_to_avoid = new_guidelines["actions_to_avoid"]
            if len(new_actions_to_avoid) >= len(game_state.guidelines["actions_to_avoid"]):
                game_state.guidelines["actions_to_avoid"] = new_actions_to_avoid


def log_game_event(episode, time_step, lives, action, reward, game_state, file_path):
    event = {
        "episode": episode,
        "time_step": time_step,
        "lives": lives,
        "action": action,
        "reward": reward,
        "game_state": game_state.to_json()  # Assuming game_state has a to_json() method
    }
    with open(file_path, 'a') as file:
        json.dump(event, file)
        file.write('\n')  # Write each event on a new line


class LLMAgent:
    def __init__(self, game_info: GameInfo):
        self.game_info = game_info
        self.current_game_state: GameState | None = None
        self.best_game_state: GameState | None = None
        self.llm = ChatOpenAI(temperature=1, model_name='gpt-3.5-turbo', max_tokens=256)

        # set up folder for logging
        base_folder = os.path.join(os.path.dirname(__file__), 'LLM')
        self.log_folder = os.path.join(base_folder, datetime.now().strftime('%Y-%m-%d_%H.%M'))
        os.makedirs(self.log_folder, exist_ok=True)
        self.init_logging()
        self.game_log = os.path.join(self.log_folder, 'game_log.jsonl')

    def init_logging(self):
        log_filename = os.path.join(self.log_folder, 'LLMAgent.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("LLM Agent logging initialized.")

    def init_game(self) -> GameState:

        if self.best_game_state is not None:
            self.current_game_state = GameState.from_game_state(self.best_game_state)
        else:
            self.current_game_state = GameState()
            self.current_game_state.available_actions = self.game_info.actions

        return self.current_game_state

    @staticmethod
    def describe_consecutive_screenshots(image, world_model, expected_entities) -> str:

        prompt_start = "This image contains 4 consecutive frames from an Atari game.\n"

        if world_model:
            prompt_world_model = f"World model: {world_model}"
        else:
            prompt_world_model = "Which game do you think it is?"

        if len(expected_entities):
            prompt_ents = f"Entities encountered previously: {', '.join(expected_entities)}.\n"
        else:
            prompt_ents = ""

        prompt_end = "Describe these four consecutive game frames individually, then summarize the overall action or motion."

        prompt = '\n'.join([prompt_start, prompt_world_model, prompt_ents, prompt_end])
        response = query_image_with_text(image, prompt)
        return response.content

    def update_best_game(self):
        if self.current_game_state is None:
            return
        if self.best_game_state is None and self.current_game_state.total_reward > 0:
            # no need to keep game states with no rewards
            self.best_game_state = self.current_game_state
        elif (self.best_game_state is not None and
              self.current_game_state.total_reward > self.best_game_state.total_reward):
            self.best_game_state = self.current_game_state

    def act(self, image, last_action, max_retries=2):
        if not self.retry_describe_screenshots(image, max_retries):
            logging.info(f'Proceeding with the last successful action: {last_action}.')
            return last_action, None, None

        llm_messages = get_llm_messages_to_update_game_state()
        llm_result = self.retry_invoke_llm(llm_messages, max_retries)

        if llm_result is None:
            logging.info(f'Proceeding with the last successful action: {last_action}.')
            return last_action, None, None

        action = update_game_state_and_act(llm_result, self.current_game_state, self.game_info)

        return action, llm_messages, llm_result

    def retry_describe_screenshots(self, image, max_retries):
        retries = 0
        while retries < max_retries:
            try:
                self.current_game_state.recent_frames_and_motion_summary = (
                    self.describe_consecutive_screenshots(image, self.current_game_state.world_model,
                                                          self.current_game_state.previously_encountered_entities))
                return True
            except Exception as e:
                retries += 1
                logging.error(f"Retry attempt {retries} failed: Error calling gpt-4-vision-preview: {e}. Retrying...")
                if retries == max_retries:
                    logging.error(
                        f"Failed after {retries} attempts: Maximum retries reached for gpt-4-vision-preview.")
                    return False

    def retry_invoke_llm(self, llm_messages, max_retries):
        retries = 0
        while retries < max_retries:
            try:
                return invoke_llm_and_parse_result(self.llm, llm_messages, self.current_game_state)
            except Exception as e:
                retries += 1
                logging.error(f"Retry attempt {retries} failed: Error calling {self.llm.model_name}: {e}. Retrying...")
                if retries == max_retries:
                    logging.error(
                        f"Failed after {retries} attempts: Maximum retries reached for {self.llm.model_name}.")
                    return None

    def update_guidelines_with_llm(self, llm_messages, llm_result, update_guidelines_message):
        ai_message = AIMessage(json.dumps(llm_result, indent=2))
        llm_messages.append(ai_message)
        llm_messages.append(update_guidelines_message)
        new_guidelines = self.retry_invoke_llm(llm_messages, max_retries=2)
        if new_guidelines is not None:
            update_guidelines(new_guidelines, self.current_game_state)

    def train(self, env, n_episodes=100, max_t=1000, save_image_interval=4):

        csv_file_path = os.path.join(self.log_folder, 'rewards.csv')

        rewards = []

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write header row
            writer.writerow(['episode', 'time_step', 'action', 'lives', 'reward', 'avg_reward', 'total_reward'])

            for i_episode in range(1, n_episodes + 1):
                # Reset the environment and preprocess the initial state
                raw_state, info = env.reset()
                # The raw state is a screen-dump of the game.
                # It is resized to 84x84 and turned to grayscale during preprocessing.
                first_frame = preprocess_frame(raw_state, self.game_info.crop_values, keep_color=True)
                frames = np.stack([first_frame] * 4, axis=0)  # Stack the initial state 4 times

                self.update_best_game()
                self.init_game()

                score = 0
                last_action = 0  # NOOP
                lives = info['lives']

                for t in range(max_t):

                    image = merge_images_with_bars(np.stack(frames, axis=0), has_color=True)
                    if t % save_image_interval == 0:
                        filename = f'4-{str(self.game_info).lower()[9:]}_{i_episode}_{t}.png'
                        save_image_to_file(image, os.path.join(self.log_folder, filename))

                    action, llm_messages, llm_result = self.act(image, last_action)

                    # execute the action and get the reward from the environment
                    next_raw_state, reward, done, truncated, info = env.step(action)
                    next_state_frame = preprocess_frame(next_raw_state, self.game_info.crop_values, keep_color=True)

                    # Update the state stack with the new frame
                    next_frames = np.append(frames[1:, :, :], np.expand_dims(next_state_frame, 0), axis=0)

                    action_text = self.game_info.actions[action]
                    self.current_game_state.recent_actions = self.current_game_state.recent_actions[1:] + [action_text]
                    self.current_game_state.recent_rewards = self.current_game_state.recent_rewards[1:] + [reward]
                    self.current_game_state.total_reward += reward

                    if llm_messages is not None:
                        if reward > 0:
                            self.update_guidelines_with_llm(llm_messages, llm_result, get_llm_message_game_reward())
                        elif info['lives'] < lives:
                            game_over = info['lives'] == 0
                            self.update_guidelines_with_llm(llm_messages, llm_result, get_llm_message_life_lost(game_over))

                    last_action = action
                    frames = next_frames
                    rewards.append(reward)
                    lives = info['lives']

                    # log results in multiple ways
                    writer.writerow([i_episode, t, action, lives, reward, round(np.mean(rewards), 2),
                                     self.current_game_state.total_reward])
                    print(f'Episode: {i_episode}  Time step: {t}  Action: {self.game_info.actions[action]}  '
                          f'Lives:  {lives}  Reward: {reward}  '
                          f'Average reward: {round(np.mean(rewards), 2)}  '
                          f'Total reward: {self.current_game_state.total_reward}')
                    log_game_event(i_episode, t, lives, action, reward, self.current_game_state, self.game_log)

                    if done or truncated:
                        break


if __name__ == '__main__':
    env_ = gym.make('BreakoutDeterministic-v4')
    agent = LLMAgent(GameInfo.BREAKOUT)
    agent.train(env_)
    env_.close()


