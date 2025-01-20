import time
from sim_class import Simulation

# Define the velocities for moving to each corner
velocities = [
    [-0.5, 0.5, 0.0],  # Move along +Y 
    [0.5, 0.0, 0.0],  # Move along +X 
    [0.0, 0.0, 0.5],  # Move along +Z 
    [-0.5, 0.0, 0.0],  # Move along -X
    [0.0, -0.5, 0.0],  # Move along -Y
    [0.5, 0.0, 0.0],  # Move along +X 
    [0.0, 0.0, -0.5],  # Move along -Z
    [-0.5, 0.0, 0.0],  # Move along -X 
]

# Initialize the simulation
sim = Simulation(num_agents=1, render=True)

for velocity in velocities:
    actions = [[velocity[0], velocity[1], velocity[2], 0]]  # Define actions for the robot

    for _ in range(300):  # Reduced steps for faster movement (fewer steps to reach each target)
        state = sim.run(actions, num_steps=5)  # Increase num_steps to speed up simulation per action
        time.sleep(0.001)  # Reduced sleep time to speed up movement

    # Extract and print the pipette position
    pipette_position = state.get('robotId_1', {}).get('pipette_position', None)
    if pipette_position:
        print(pipette_position)

# positions are printed in the terminal when running the code
# [-0.187, 0.2195, 0.1195]
# [0.253, 0.2195, 0.1195]
# [0.2529, 0.2195, 0.2895]
#[-0.187, 0.2195, 0.2895]
# [-0.1857, -0.1705, 0.2894]
# [0.253, -0.1705, 0.2894]
# [0.25, -0.1705, 0.1694]
#[-0.187, -0.1705, 0.1695]