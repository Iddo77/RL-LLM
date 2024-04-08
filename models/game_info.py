from enum import Enum


actions = [
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE"
        ]

class GameInfo(Enum):
    BOXING = {"crop_values": (13, 97), "actions": actions[:]}
    BREAKOUT = {"crop_values": (18, 102), "actions": ["NOOP", "FIRE", "RIGHT", "LEFT",]}
    RIVERRAID = {"crop_values": (2, 86), "actions": actions[:]}
    PONG = {"crop_values": (17, 101), "actions": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]}

    @property
    def crop_values(self):
        return self.value["crop_values"]

    @property
    def actions(self):
        return self.value["actions"]
