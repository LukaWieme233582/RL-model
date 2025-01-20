import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2GymWrapper(gym.Env):
    def __init__(self, render=False):
        super(OT2GymWrapper, self).__init__()
        
        # Action and observation space setup (adjust as needed)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Initialize the simulation with the render option
        self.sim = Simulation(num_agents=1, render=render)

        # Define the target position
        self.target_position = np.array([0.5, 0.5, 0.5])  # Example target

        # Other attributes
        self.current_state = self._get_pipette_position()
        self.tolerance = 0.05  # Distance threshold for "done"

    def reset(self, target_position=None):
        # Reset the simulation
        self.sim.reset()
        
        # Update the target position (randomize if not provided)
        if target_position is None:
            self.target_position = np.random.uniform(low=0, high=1, size=(3,))
        else:
            self.target_position = np.array(target_position)
        
        self.current_state = self._get_pipette_position()
        return np.array(self.current_state, dtype=np.float32)

    def step(self, action):
        velocity = [action[0], action[1], action[2], 0]
        actions = [velocity]
        self.sim.run(actions, num_steps=5)
        self.current_state = self._get_pipette_position()
        reward = self._calculate_reward()
        done = self._is_done()
        return np.array(self.current_state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        """
        Render the environment (if rendering is enabled in the simulation).
        """
        pass  # Rendering handled by Simulation class if enabled

    def close(self):
        """
        Clean up resources used by the environment.
        """
        self.sim.close()

    def _get_pipette_position(self):
        """
        Retrieves the pipette position from the simulation state.
        """
        state = self.sim.get_states()  # Get the latest state from the simulation
        pipette_position = state.get('robotId_2', {}).get('pipette_position', [0, 0, 0])
        return pipette_position

    def _calculate_reward(self):
        distance = np.linalg.norm(np.array(self.current_state) - np.array(self.target_position))
        max_distance = np.linalg.norm(np.array([1, 1, 1]) - np.array([0, 0, 0]))  # Assuming normalized space (0 to 1)
        reward = 1 - (distance / max_distance)  # Normalize and invert distance
        return max(0, reward)  # Ensure reward stays within [0, 1]


    def _is_done(self):
        distance = np.linalg.norm(np.array(self.current_state) - np.array(self.target_position))
        return distance < self.tolerance
