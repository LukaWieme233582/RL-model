import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation xxenvironment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([-1.927, -1.9105, -1.8805, -1.927, -1.9105, -1.8805]),
                                            high=np.array([2.073, 2.0895, 2.1195, 2.073, 2.0895, 2.1195]),
                                            dtype=np.float32)

        # keep track of the number of steps
        self.steps = 0
        self.previouse_distance = None

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)

        self.goal_position = np.random.uniform(
        low=[-0.187, -0.1705, 0.1195],
        high=[0.253, 0.2195, 0.2895], size=3)
    

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)
        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32

        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)
 

    # Append the goal position to the pipette position
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Reset the number of steps
        self.steps = 0

        return observation, {}

    def step(self, action):
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Calculate the reward, this is something that you will need to experiment with to get the best results


        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        reward = -distance_to_goal * 0.1
        
        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        if distance_to_goal < 0.001:
            terminated = True
            # we can also give the agent a positive reward for completing the task
            reward += 40
        else:
            terminated = False

        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {} # we don't need to return any additional information

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
