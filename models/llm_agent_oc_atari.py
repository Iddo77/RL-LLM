import json
import os
import logging
from datetime import datetime

import numpy as np
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from ocatari.core import OCAtari

from models.game_info import GameInfo
from models.agent_state_oc_atari import AgentState
from image_processing import preprocess_frame, merge_images_with_bars, save_image_to_file
from models.game_logger import GameLogger
from utils import parse_json_from_substring, escape_brackets, trim_list


# and environment variable OPENAI_API_KEY must be set with the OpenAI key

def get_llm_messages_to_update_agent_state():
    prompt_text = """### AGENT STATE
{game_state}

### INSTRUCTIONS
- Describe a world model in a few short sentences (you can keep the old one if it is fine).
- Describe the current game state in a single sentence. It will be appended to the recent_state_descriptions of the agent state.
- Describe the transition that occurs between the previous and the current game state as a motion in a single sentence. It will be appended to the recent_motion_descriptions of the agent state.
- Give the next action, by choosing one from the available actions. Try to vary your actions if you don't know what to do.

### RESULT
Create a json containing the world model, the state and motion description, and the next action. Write nothing else. For example:
{
    "world_model": "A game to hunt ducks"
    "state_description": "A duck is on the left and the player is at the bottom in the middle of the screen."
    "motion_description": "A duck is moving horizontally from right to left",
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


def update_game_state_and_act(llm_result: dict, game_state: AgentState, game_info: GameInfo):

    if "world_model" in llm_result:
        game_state.world_model = llm_result["world_model"]
    if "state_description" in llm_result:
        game_state.recent_state_descriptions.append(llm_result["state_description"])
        trim_list(game_state.recent_state_descriptions)
    if "motion_description" in llm_result:
        game_state.recent_motion_descriptions.append(llm_result["motion_description"])
        trim_list(game_state.recent_motion_descriptions)

    action = 0  # NOOP
    if "next_action" in llm_result:
        action_text = llm_result["next_action"]
        if action_text in game_info.actions:
            action = game_info.actions.index(action_text)

    return action


def update_guidelines(llm_result: dict, game_state: AgentState):
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


class LLMAgentOcAtari:
    def __init__(self, game_info: GameInfo):
        self.game_info = game_info
        self.current_agent_state: AgentState | None = None
        self.best_agent_state: AgentState | None = None
        self.llm = ChatOpenAI(temperature=1, model_name='gpt-3.5-turbo', max_tokens=256)
        # the guide is separate, because it can produce more output tokens
        self.llm_guide = ChatOpenAI(temperature=1, model_name='gpt-4-turbo-preview', max_tokens=512)
        self.game_logger: GameLogger | None = None

    def init_game(self) -> AgentState:

        if self.best_agent_state is not None:
            self.current_agent_state = AgentState.from_agent_state(self.best_agent_state)
        else:
            self.current_agent_state = AgentState()
            self.current_agent_state.available_actions = self.game_info.actions

        self.current_agent_state.previous_game_state = "Missing. The game just started"

        return self.current_agent_state

    def update_best_agent_state(self):
        if self.current_agent_state is None:
            return
        if self.best_agent_state is None and self.current_agent_state.total_reward > 0:
            # no need to keep game states with no rewards
            self.best_agent_state = self.current_agent_state
        elif (self.best_agent_state is not None and
              self.current_agent_state.total_reward > self.best_agent_state.total_reward):
            self.best_agent_state = self.current_agent_state

    def act(self, last_action, episode, time_step):

        llm_messages = get_llm_messages_to_update_agent_state()
        llm_result = self.retry_invoke_llm(self.llm, llm_messages, max_retries=2, episode=episode, time_step=time_step)
        llm_result = parse_json_from_substring(llm_result)

        if llm_result is None:
            return last_action, None, None

        action = update_game_state_and_act(llm_result, self.current_agent_state, self.game_info)

        return action, llm_messages, llm_result

    def retry_invoke_llm(self, llm, llm_messages, max_retries, episode, time_step):
        retries = 0
        while retries < max_retries:
            try:
                chat_prompt_template = ChatPromptTemplate.from_messages(llm_messages)
                chain = LLMChain(llm=llm, prompt=chat_prompt_template)
                result = chain.invoke({"game_state": self.current_agent_state.to_json()})
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
            update_guidelines(new_guidelines, self.current_agent_state)

    def train(self, env, max_episodes=5, max_time_steps=2000, save_image_interval=4):

        self.game_logger = GameLogger('LLM-Agent-OcAtari', self.game_info)
        total_time_steps = 0

        for i_episode in range(1, max_episodes + 1):
            # Reset the environment and preprocess the initial state
            raw_state, info = env.reset()

            # The raw state is a screen-dump of the game.
            # It is resized to 84x84 and turned to grayscale during preprocessing.
            # It is not used to train the agent, only for visual logging of the game state.
            first_frame = preprocess_frame(raw_state, self.game_info.crop_values, keep_color=True)
            frames = np.stack([first_frame] * 4, axis=0)  # Stack the initial state 4 times

            self.update_best_agent_state()
            self.init_game()

            last_action = 0  # NOOP
            t = 0
            lives = info.get('lives', 0)
            game_over = False
            score = 0

            while not game_over:
                if t % save_image_interval == 0:
                    image = merge_images_with_bars(np.stack(frames, axis=0), has_color=True)
                    filename = f'4-{str(self.game_info).lower()[9:]}_{i_episode}_{t}.png'
                    save_image_to_file(image, os.path.join(self.game_logger.log_folder, filename))

                self.current_agent_state.current_game_state = str(env.objects)

                action, llm_messages, llm_result = self.act(last_action, i_episode, t)

                # execute the action and get the reward from the environment
                next_raw_state, reward, done, truncated, info = env.step(action)

                # update frames
                next_state_frame = preprocess_frame(next_raw_state, self.game_info.crop_values, keep_color=True)
                frames = np.append(frames[1:, :, :], np.expand_dims(next_state_frame, 0), axis=0)

                action_text = self.game_info.actions[action]
                self.current_agent_state.recent_actions.append(action_text)
                trim_list(self.current_agent_state.recent_actions)
                self.current_agent_state.recent_rewards.append(reward)
                trim_list(self.current_agent_state.recent_rewards)
                self.current_agent_state.total_reward += reward

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
                score += reward
                lives = info.get('lives', 0)
                total_time_steps += 1  # total time-steps so far
                t += 1  # time-steps in this episode
                game_over = done or truncated

                self.game_logger.log_game_data(i_episode, t, action, lives, reward, score, self.current_agent_state)

                # move current game state to previous (done after logging on purpose)
                self.current_agent_state.previous_game_state = self.current_agent_state.current_game_state
                self.current_agent_state.current_game_state = ""

                if total_time_steps >= max_time_steps:
                    return

        self.game_logger.close()


if __name__ == '__main__':
    start_time = datetime.now()
    print(f"Start Time: {start_time}")
    env_ = OCAtari("BreakoutDeterministic-v4", mode="ram", hud=False, render_mode="rgb_array")
    agent = LLMAgentOcAtari(GameInfo.BREAKOUT)
    agent.train(env_)
    env_.close()
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"End Time: {end_time}")
    print(f"Duration: {duration}")
