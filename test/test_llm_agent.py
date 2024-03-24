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


if __name__ == '__main__':
    unittest.main()
