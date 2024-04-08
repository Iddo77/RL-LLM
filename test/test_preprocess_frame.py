import unittest
import gymnasium as gym
import numpy as np

from models.game_info import GameInfo
from image_processing import preprocess_frame, convert_to_grayscale, resize_frame, merge_images_with_bars, save_image_to_file


class TestPreprocessFrame(unittest.TestCase):
    def test_no_preprocessing(self):

        env = gym.make('BoxingNoFrameskip-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(8):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            save_image_to_file(next_state, f'BoxingNoPreprocess{i + 1}.png')
            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_grayscale(self):

        env = gym.make('BoxingNoFrameskip-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(8):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = convert_to_grayscale(next_state)
            save_image_to_file(next_state, f'BoxingGray{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_frameskip(self):

        env = gym.make('BoxingDeterministic-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(4):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            save_image_to_file(next_state, f'BoxingFrameSkip{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_resize(self):

        env = gym.make('BoxingDeterministic-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(4):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = resize_frame(next_state, 84, 110)
            save_image_to_file(next_state, f'BoxingResize{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_resize_and_crop(self):

        env = gym.make('BoxingDeterministic-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(4):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = resize_frame(next_state, 84, 110)
            next_state = next_state[18:102, :]
            save_image_to_file(next_state, f'BoxingResizeAndCrop{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_preprocess_boxing(self):
        env = gym.make('BoxingDeterministic-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(4):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_frame(next_state, GameInfo.BOXING.crop_values)
            save_image_to_file(next_state, f'Boxing{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_preprocess_breakout(self):
        env = gym.make('BreakoutDeterministic-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(4):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_frame(next_state, GameInfo.BREAKOUT.crop_values)
            save_image_to_file(next_state, f'Breakout{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_preprocess_riverraid(self):
        env = gym.make('RiverraidDeterministic-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(28):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            if i < 20:
                continue
            next_state = preprocess_frame(next_state, GameInfo.RIVERRAID.crop_values)
            save_image_to_file(next_state, f'Riverraid{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

    def test_merge_images_with_bars(self):
        env = gym.make('RiverraidDeterministic-v4')
        env.reset()

        frames = []

        # Process and save the first 8 frames
        for i in range(28):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            if i < 20:
                continue
            next_state = preprocess_frame(next_state, GameInfo.RIVERRAID.crop_values)
            frames.append(next_state)
            if len(frames) == 4:
                break
            if terminated or truncated:
                break

        env.close()

        image_array = np.stack(frames, axis=0)
        image = merge_images_with_bars(image_array)
        save_image_to_file(image, '4-images.png')

        self.assertTrue(True)

    def test_merge_images_with_bars_breakout(self):
        env = gym.make('BreakoutDeterministic-v4')
        env.reset()

        frames = []

        for i in range(4):
            if i == 0:
                action = 1  # FIRE
            else:
                action = 2  # RIGHT

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_frame(next_state, GameInfo.BREAKOUT.crop_values)
            frames.append(next_state)
            if terminated or truncated:
                break

        env.close()

        image_array = np.stack(frames, axis=0)
        image = merge_images_with_bars(image_array)
        save_image_to_file(image, '4-breakout.png')

        self.assertTrue(True)


    def test_pong(self):

        env = gym.make('PongDeterministic-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(4):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_frame(next_state, GameInfo.PONG.crop_values)
            save_image_to_file(next_state, f'PongDeterministic{i + 1}.png')

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
