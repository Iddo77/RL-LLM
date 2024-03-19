import gymnasium as gym
import numpy as np
import os
from transformers import pipeline

# Load a text generation pipeline
generator = pipeline('text-generation', model='stabilityai/stablelm-zephyr-3b')

# Mock LLM function
# def mock_llm_response(prompt):
#     # call to LLM
#     responses = ["fire the main engine", "move left", "move right", "do nothing"]
#     return np.random.choice(responses)

def llm_response(prompt):
    try:
        responses = generator(prompt, max_length=1000, do_sample=True, temperature=0.5)
        text_response = responses[0]['generated_text'].strip()
        return text_response
    except Exception as e:
        print(f"Error in LLM response: {e}")
        return "do nothing" # fallback

def state_to_text(state):
    print(f"State received: {state}")
    state_array, _ = state
    position_x, position_y, velocity_x, velocity_y, angle, angular_velocity, left_leg_contact, right_leg_contact = state_array
    return f"""Lunar lander is located at ({position_x:.2f}, {position_y:.2f}) 
            with velocities in the x and y directions being {velocity_x:.2f} and {velocity_y:.2f}, respectively. 
            The lander is tilted at an angle of {angle:.2f} degrees with an angular velocity of {angular_velocity:.2f}. 
            The left leg contact is {int(left_leg_contact)} and the right leg contact is {int(right_leg_contact)}."""

response_to_action = {
    "fire the main engine": 2,
    "move left": 1,
    "move right": 3,
    "do nothing": 0
}

# Initialize environment
env = gym.make('LunarLander-v2')

# Run
initial_state = env.reset()
print(f"Initial state shape: {initial_state[0].shape}")
state = initial_state
done = False
total_reward = 0

while not done:
    env.render()
    text_state = state_to_text(state)
    prompt = f"Given the current state of the lunar lander described as follows:\n{text_state}\nThe goal is to land the lander between the flags softly without crashing. What should the lunar lander do next?"
    llm_text_response = llm_response(prompt)
    print(f"LLM Response: {llm_text_response}")
    action = response_to_action.get(llm_text_response.lower(), 0)
    state, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Total Reward: {total_reward}")
env.close()