import unittest
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import preprocess_frame, convert_to_grayscale, resize_frame, CropValues


class TestPreprocessFrame(unittest.TestCase):
    def test_no_preprocessing(self):

        env = gym.make('BoxingNoFrameskip-v4')
        env.reset()

        # Process and save the first 8 frames
        for i in range(8):
            action = env.action_space.sample()  # Taking a random action
            next_state, reward, terminated, truncated, info = env.step(action)
            create_and_save_image(next_state, i + 1, "BoxingNoPreprocess")
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
            create_and_save_image(next_state, i + 1, "BoxingGray")

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
            create_and_save_image(next_state, i + 1, "BoxingFrameSkip")

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
            create_and_save_image(next_state, i + 1, "BoxingResize")

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
            create_and_save_image(next_state, i + 1, "BoxingResizeAndCrop")

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
            next_state = preprocess_frame(next_state, CropValues.BOXING)
            create_and_save_image(next_state, i + 1, "Boxing")

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
            next_state = preprocess_frame(next_state, CropValues.BREAKOUT)
            create_and_save_image(next_state, i + 1, "Breakout")

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
            next_state = preprocess_frame(next_state, CropValues.RIVERRAID)
            create_and_save_image(next_state, i + 1, "Riverraid")

            if terminated or truncated:
                break

        env.close()
        self.assertTrue(True)


def create_and_save_image(state, frame_number, name):
    plt.imshow(state, cmap='gray')
    plt.title(f'Frame {frame_number}')
    plt.axis('off')
    plt.savefig(f'{name}{frame_number}.png')
    plt.close()


if __name__ == '__main__':
    unittest.main()
