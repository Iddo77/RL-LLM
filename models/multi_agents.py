from pettingzoo.atari import boxing_v2

env = boxing_v2.env(render_mode="human")
observation, info = env.reset()

for episode in range(20):  # Run 20 episodes
    observation = env.reset() 
    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

env.close()