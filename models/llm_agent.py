import json
import os
import numpy as np
import gymnasium as gym
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

from models.game_info import GameInfo
from models.game_state import GameState
from utils import preprocess_frame, convert_image_to_base64, merge_images_with_bars, parse_json_from_substring


# and environment variable OPENAI_API_KEY must be set with the OpenAI key


def query_image_with_text(image, text):

    base64_image = convert_image_to_base64(image)
    chat = ChatOpenAI(model='gpt-4-vision-preview', max_tokens=512)
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
- Describe the entities in the current frame state and their positions. Use existing entities as much as possible. 
- Describe a world model in a few short sentences.
- Give the next action, by choosing one from the available actions. Try to vary your actions if you don't know what to do.

### RESULT
Create a json containing entities_in_game_state and next_action. Write nothing else. For example:
{
    "entities_in_game_state": [{"name": "duck", "position": "bottom-left"}],
   "world_model": "A game to hunt ducks"
    "next_action": "FIRE"
}"""

    # escape brackets in json, otherwise validation of langchain will fail
    prompt_text = prompt_text.replace('{', '{{')
    prompt_text = prompt_text.replace('}', '}}')
    prompt_text = prompt_text.replace('{{game_state}}', '{game_state}')

    system_message = SystemMessage("You are an RL agent playing a game.")
    prompt_template = HumanMessagePromptTemplate.from_template(input_variables=["game_state"], template=prompt_text)
    return [system_message, prompt_template]


def get_llm_message_game_over():
    prompt_text = """Game over!!! 
    
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
    return HumanMessagePromptTemplate.from_template(template=prompt_text)


def invoke_llm_and_parse_result(llm: ChatOpenAI,
                                llm_messages: list[SystemMessage | HumanMessagePromptTemplate | AIMessage],
                                game_state: GameState):
    chat_prompt_template = ChatPromptTemplate.from_messages(llm_messages)
    chain = LLMChain(llm=llm, prompt=chat_prompt_template)
    result = chain.invoke({"game_state": game_state.to_json()})
    return parse_json_from_substring(result["text"])


def update_game_state_and_act(llm_result: dict, game_state: GameState, game_info: GameInfo):

    if "entities_in_game_state" in llm_result:
        valid_entities = []
        entities = llm_result["entities_in_game_state"]
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    name = ent.get("name")
                    position = ent.get("position")
                    if name is not None and position is not None:
                        valid_entities.append(ent)
                        if name not in game_state.entities_encountered:
                            game_state.entities_encountered.append(name)

    if "world_model" in llm_result:
        game_state.world_model = llm_result["world_model"]

    action = 0  # NOOP
    if "next_action" in llm_result:
        action_text = llm_result["next_action"]
        if action_text in game_info.actions:
            action = game_info.actions.index(action_text)
    else:
        # hack to make sure the llm-result can be reused later with the action
        llm_result["next_action"] = "NOOP"

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


class LLMAgent:
    def __init__(self, game_info: GameInfo):
        self.game_info = game_info
        self.current_game_state: GameState | None = None
        self.best_game_state: GameState | None = None
        self.llm = ChatOpenAI(temperature=1, model_name='gpt-3.5-turbo', max_tokens=256)

    def init_game(self) -> GameState:

        if self.best_game_state is not None:
            self.current_game_state = GameState.from_game_state(self.best_game_state)
        else:
            self.current_game_state = GameState()
            self.current_game_state.available_actions = self.game_info.actions

        return self.current_game_state

    @staticmethod
    def describe_frames(frames) -> str:
        image_stack = np.stack(frames, axis=0)
        image = merge_images_with_bars(image_stack)
        text = """Describe these four consecutive game frames individually, then summarize the overall action or motion.
         Do not write anything else."""
        response = query_image_with_text(image, text)
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

    def train(self, env, n_episodes=100, max_t=1000, save_interval=100, log_interval=10, state_dir='LLM/state'):
        scores = []
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)

        for i_episode in range(1, n_episodes + 1):
            # Reset the environment and preprocess the initial state
            raw_state, info = env.reset()
            # The raw state is a screen-dump of the game.
            # It is resized to 84x84 and turned to grayscale during preprocessing.
            first_frame = preprocess_frame(raw_state, self.game_info.crop_values)
            frames = np.stack([first_frame] * 4, axis=0)  # Stack the initial state 4 times

            self.update_best_game()
            self.init_game()

            score = 0
            for t in range(max_t):

                self.current_game_state.game_state_description = self.describe_frames(frames)

                llm_messages = get_llm_messages_to_update_game_state()
                llm_result = invoke_llm_and_parse_result(self.llm, llm_messages, self.current_game_state)
                action = update_game_state_and_act(llm_result, self.current_game_state, self.game_info)

                # execute the action and get the reward from the environment
                next_raw_state, reward, done, truncated, info = env.step(action)
                next_state_frame = preprocess_frame(next_raw_state, self.game_info.crop_values)
                # Update the state stack with the new frame
                next_frames = np.append(frames[1:, :, :], np.expand_dims(next_state_frame, 0), axis=0)

                action_text = self.game_info.actions[action]
                self.current_game_state.recent_actions = self.current_game_state.recent_actions[1:] + [action_text]
                self.current_game_state.recent_rewards = self.current_game_state.recent_rewards[1:] + [reward]
                self.current_game_state.total_reward += reward

                if reward > 0:
                    ai_message = AIMessage(json.dumps(llm_result, indent=2))
                    llm_messages.append(ai_message)
                    llm_messages.append(get_llm_message_game_reward())
                    llm_result_reward = invoke_llm_and_parse_result(self.llm, llm_messages, self.current_game_state)
                    update_guidelines(llm_result_reward, self.current_game_state)
                    pass

                frames = next_frames
                score += reward

                if done or truncated:
                    ai_message = AIMessage(json.dumps(llm_result, indent=2))
                    llm_messages.append(ai_message)
                    llm_messages.append(get_llm_message_game_over())
                    llm_result_game_over = invoke_llm_and_parse_result(self.llm, llm_messages, self.current_game_state)
                    update_guidelines(llm_result_game_over, self.current_game_state)
                    break

            scores.append(score)

            if i_episode % log_interval == 0:
                average_score = np.mean(scores[-log_interval:])
                print(f"Episode {i_episode}/{n_episodes} - Average Score: {average_score}")

        env.close()
        return scores


if __name__ == '__main__':
    env = gym.make('BreakoutNoFrameskip-v4')
    agent = LLMAgent(GameInfo.BREAKOUT)
    scores = agent.train(env)
    env.close()


