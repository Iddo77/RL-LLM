import json
import os
import logging
from datetime import datetime

import numpy as np
import gymnasium as gym
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

from models.game_info import GameInfo
from models.game_logger import GameLogger
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


def get_update_guidelines_message(prefix):
    prompt_text = prefix + """You should update the guidelines for what to do in the future. 
For example: Choose LEFT if the crosshair is to the right of the target.

Respond with a json like this:

   {
       "guidelines": {
            "recommendations": a list of short recommendations about what to do,
            "things_to_avoid": a list of short rules about what not to do
        }
   }

Keep existing guidelines as much as possible, but make sure the lists do not exceed 10 rules each.

Respond with the json only and write nothing else.
"""

    # escape brackets in json, otherwise validation of langchain will fail
    prompt_text = escape_brackets(prompt_text, [])

    return HumanMessagePromptTemplate.from_template(template=prompt_text)


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
        if "recommendations" in new_guidelines:
            new_recommendations = new_guidelines["recommendations"]
            if len(new_recommendations) >= len(game_state.guidelines["recommendations"]):
                game_state.guidelines["recommendations"] = new_recommendations
        if "things_to_avoid" in new_guidelines:
            new_things_to_avoid = new_guidelines["things_to_avoid"]
            if len(new_things_to_avoid) >= len(game_state.guidelines["things_to_avoid"]):
                game_state.guidelines["things_to_avoid"] = new_things_to_avoid


class LLMVisionAgent:
    def __init__(self, game_info: GameInfo):
        self.game_info = game_info
        self.current_game_state: GameState | None = None
        self.best_game_state: GameState | None = None
        self.llm = ChatOpenAI(temperature=1, model_name='gpt-3.5-turbo', max_tokens=256)
        # GPT 3.5 is not smart enough to update guidelines, so GPT-4 is used for that
        self.llm_guide = ChatOpenAI(temperature=1, model_name='gpt-4-turbo-preview', max_tokens=512)
        self.scores = []
        self.game_logger: GameLogger | None = None

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

    def act(self, image, last_action, episode, time_step):
        if not self.retry_describe_screenshots(image, max_retries=2):
            return last_action, None, None

        llm_messages = get_llm_messages_to_update_game_state()
        llm_result = self.retry_invoke_llm(self.llm, llm_messages, max_retries=2, episode=episode, time_step=time_step)
        llm_result = parse_json_from_substring(llm_result)

        if llm_result is None:
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

    def retry_invoke_llm(self, llm, llm_messages, max_retries, episode, time_step):
        retries = 0
        while retries < max_retries:
            try:
                chat_prompt_template = ChatPromptTemplate.from_messages(llm_messages)
                chain = LLMChain(llm=llm, prompt=chat_prompt_template)
                result = chain.invoke({"game_state": self.current_game_state.to_json()})
                self.game_logger.log_llm_result(episode, time_step, result["text"])
                return result["text"]
            except Exception as e:
                retries += 1
                logging.error(f"Episode: {episode}  Time step: {time_step}  "
                              f"Retry attempt {retries} failed: Error calling {llm.model_name}: {e}. Retrying...")
                if retries == max_retries:
                    logging.error(
                        f"Failed after {retries} attempts: Maximum retries reached for {llm.model_name}.")
                    return None

    def update_guidelines_with_llm(self, llm_messages, llm_result, update_guidelines_message, episode, time_step):
        ai_message = AIMessage(json.dumps(llm_result, indent=2))
        llm_messages.append(ai_message)
        llm_messages.append(update_guidelines_message)
        llm_result = self.retry_invoke_llm(self.llm_guide, llm_messages, max_retries=2,
                                           episode=episode, time_step=time_step)
        new_guidelines = parse_json_from_substring(llm_result)
        if new_guidelines is not None:
            update_guidelines(new_guidelines, self.current_game_state)

    def train(self, env, max_episodes=5, max_total_time_steps=2000, max_time_steps_per_episode=500,
              save_image_interval=4):

        self.game_logger = GameLogger('LLM-Vision-Agent', self.game_info)
        total_time_steps = 0

        for i_episode in range(1, max_episodes + 1):
            # Reset the environment and preprocess the initial state
            raw_state, info = env.reset()
            # The raw state is a screen-dump of the game.
            # It is resized to 84x84 and turned to grayscale during preprocessing.
            first_frame = preprocess_frame(raw_state, self.game_info.crop_values, keep_color=True)
            frames = np.stack([first_frame] * 4, axis=0)  # Stack the initial state 4 times

            self.update_best_game()
            self.init_game()

            last_action = 0  # NOOP
            t = 0
            lives = info.get('lives', 0)
            game_over = False
            score = 0

            while not game_over:

                image = merge_images_with_bars(np.stack(frames, axis=0), has_color=True)
                if t % save_image_interval == 0:
                    filename = f'4-{str(self.game_info).lower()[9:]}_{i_episode}_{t}.png'
                    save_image_to_file(image, os.path.join(self.game_logger.log_folder, filename))

                action, llm_messages, llm_result = self.act(image, last_action, i_episode, t)

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
                        prefix = ('You received a reward following your last action! '
                                  'That means you did something right.\n\n')
                        self.update_guidelines_with_llm(llm_messages, llm_result,
                                                        get_update_guidelines_message(prefix), i_episode, t)
                    elif reward < 0:
                        prefix = ('You received a penalty following your last action! '
                                  'That means you did something wrong.\n\n')
                        self.update_guidelines_with_llm(llm_messages, llm_result,
                                                        get_update_guidelines_message(prefix), i_episode, t)
                    elif info.get('lives', 0) < lives:
                        prefix = 'You lost a life!\n\n'
                        self.update_guidelines_with_llm(llm_messages, llm_result,
                                                        get_update_guidelines_message(prefix), i_episode, t)

                last_action = action
                frames = next_frames
                score += reward
                lives = info.get('lives', 0)
                total_time_steps += 1  # total time-steps so far
                t += 1  # time-steps in this episode
                game_over = done or truncated

                self.game_logger.log_game_data(i_episode, t, action, lives, reward, score, self.current_game_state)

                if t >= max_time_steps_per_episode:
                    break
                if total_time_steps >= max_total_time_steps:
                    self.game_logger.close()
                    return

        self.game_logger.close()


if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    env_ = gym.make('BoxingDeterministic-v4')
    agent = LLMVisionAgent(GameInfo.BOXING)
    agent.train(env_)
    env_.close()
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"End Time: {end_time}")
    print(f"Duration: {duration}")

