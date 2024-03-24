import json


class GameState:
    def __init__(self):
        self.game_state_description = ""
        self.entities_encountered = []
        self.entities_in_game_state = []
        self.available_actions = []
        self.recent_actions = ['NOOP'] * 4
        self.recent_rewards = [0, 0, 0, 0]
        self.guidelines = {
            "recommended_actions": [],
            "actions_to_avoid": []
        }
        self.total_reward = 0
        self.next_action = ""

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
        """Creates a new GameState instance, retaining only entities_encountered, guidelines and
        available_actions from the original."""
        new_instance = cls()
        new_instance.entities_encountered = game_state.entities_encountered.copy()
        new_instance.guidelines = game_state.guidelines.copy()
        new_instance.available_actions = game_state.available_actions
        return new_instance
