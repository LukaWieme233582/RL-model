import gymnasium as gym
import numpy as np

from ot2_gym_wrapper_2 import OT2Env

# Create and configure the environment
environment = OT2Env(render=False, max_steps=1000)

# Run a single episode
observation = environment.reset()
is_done = False
steps_taken = 0

while not is_done:
    # Select a random action from the action space
    selected_action = environment.action_space.sample()
    
    # Perform the action in the environment
    observation, reward, terminated, truncated, info = environment.step(selected_action)

    print(f"Step: {steps_taken + 1}, Action: {selected_action}, Reward: {reward}")

    steps_taken += 1
    is_done = terminated or truncated

    if is_done:
        print(f"Episode ended after {steps_taken} steps. Info: {info}")
        break
