import json


class AgentState:
    def __init__(self):
        self.world_model = ""
        self.available_actions = []
        self.previous_game_state = ""
        self.current_game_state = ""
        self.recent_motion_descriptions = []
        self.recent_state_descriptions = []
        self.recent_actions = []
        self.recent_rewards = []
        self.guidelines = {
            "recommendations": [],
            "things_to_avoid": []
        }
        self.total_reward = 0.0

    def to_json(self):
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, json_string):
        data = json.loads(json_string)
        game_state = cls()
        game_state.__dict__.update(data)
        return game_state

    @classmethod
    def from_agent_state(cls, game_state: 'AgentState'):
        """Creates a new AgentState instance, retaining only previously_encountered_entities, guidelines and
        available_actions from the original."""
        new_instance = cls()
        new_instance.world_model = game_state.world_model
        new_instance.guidelines = game_state.guidelines.copy()
        new_instance.available_actions = game_state.available_actions
        return new_instance
