import unittest
import numpy as np
from PIL import Image

from models.llm_agent import query_image_with_text


class TestLLMAgent(unittest.TestCase):
    def test_query_image_with_text(self):
        image_array = np.array(Image.open('4-images.png'))
        text = """Describe these four consecutive game frames individually, then summarize the overall action or motion.
         Do not write anything else."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_riverraid(self):
        image_array = np.array(Image.open('4-images.png'))
        text = """Describe these 4 consecutive frames of the Atari game RiverRaid individually, then summarize the overall action or motion.
         Do not write anything else."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_1(self):
        image_array = np.array(Image.open('4-breakout.png'))
        text = """Describe these four consecutive game frames individually, then summarize the overall action or motion.
         Do not write anything else."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_2(self):
        image_array = np.array(Image.open('4-breakout.png'))
        text = """Describe these four consecutive frames of the Atari game Breakout individually, then summarize the overall action or motion.
         Do not write anything else."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_color(self):
        image_array = np.array(Image.open('4-breakout-color.png'))
        text = """This image contains 4 consecutive screenshots from an Atari game. Which game do you think it is? 
        And what objects are visible and in what position? 
        Respond with a json with 'world_model': the description of the game, "entities": a list of entities. 
        Each entity is an object with 'name', 'final-position' and 'motion'.
        For example {'name': 'player', 'final-position': 'bottom-left, to the left of the treasure chest', 'motion': 'moving to the left'}"""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_color_2(self):
        image_array = np.array(Image.open('4-breakout-color.png'))
        text = """This image contains 4 consecutive screenshots from an Atari game. Which game do you think it is? 
        And what objects are visible and in what position? """
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_color_3(self):
        image_array = np.array(Image.open('4-breakout-color.png'))
        text = """This image is contains 4 consecutive screenshots from an Atari game. Which game do you think it is? 
        Describe these four consecutive game screenshots individually, then summarize the overall action or motion."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_3(self):
        image_array = np.array(Image.open('4-breakout.png'))
        text = """This image is contains 4 consecutive screenshots from an Atari game. Which game do you think it is? 
        Describe these four consecutive game screenshots individually, then summarize the overall action or motion."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_color_4(self):
        image_array = np.array(Image.open('4-breakout-color.png'))
        text = """This image contains 4 consecutive screenshots from an Atari game. Which game do you think it is? 
        And what objects are visible and in what position? 
        Respond with 1 object per frame per line
        Respond with 'world_model': the description of the game, "entities": a list of entities. 
        Each entity is an object with 'name', 'final-position' and 'motion'.
        For example {'name': 'player', 'final-position': 'bottom-left, to the left of the treasure chest', 'motion': 'moving to the left'}"""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_color_5(self):
        image_array = np.array(Image.open('4-breakout-color.png'))
        text = """This image is contains 4 consecutive frames from an Atari game. Which game do you think it is? 
        Describe these 4 consecutive game frames individually, then summarize the overall action or motion."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)

    def test_breakout_color_6(self):
        image_array = np.array(Image.open('4-breakout-color.png'))
        text = """### CONTEXT
        {
"world_model": "A classic Breakout game where the player controls a paddle to bounce a ball towards colored bricks at the top of the screen.",
"entities": [
{"name": "ball", "position": "close to colored bricks", "motion": "moving upwards"},
{"name": "paddle", "position": "near the bottom", "motion": "stationary"},
{"name": "colored bricks", "position": "at the top of the screen", "motion": "stationary"}
]
}

### IMAGE
This image is contains 4 consecutive frames from the Atari game described above.

## INSTRUCTIONS
Describe these 4 consecutive game frames individually, then summarize the overall action or motion."""
        response = query_image_with_text(image_array, text)
        print(response.content)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
