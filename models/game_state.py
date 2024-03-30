import json


class GameState:
    def __init__(self):
        self.world_model = ""
        self.recent_frames_and_motion_summary = ""
        self.previously_encountered_entities = []
        self.entities_in_previous_game_state = []
        self.available_actions = []
        self.recent_actions = ['NOOP'] * 4
        self.recent_rewards = [0.0] * 4
        self.guidelines = {
            "recommended_actions": [],
            "actions_to_avoid": []
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
    def from_game_state(cls, game_state: 'GameState'):
        """Creates a new GameState instance, retaining only previously_encountered_entities, guidelines and
        available_actions from the original."""
        new_instance = cls()
        new_instance.previously_encountered_entities = game_state.previously_encountered_entities.copy()
        new_instance.guidelines = game_state.guidelines.copy()
        new_instance.available_actions = game_state.available_actions
        return new_instance
