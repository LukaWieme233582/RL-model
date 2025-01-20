import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, target_position):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_position = target_position
        self.prev_error = np.array([0.0, 0.0, 0.0])  # 3D error for each axis
        self.integral = np.array([0.0, 0.0, 0.0])  # 3D integral for each axis

    def update_position(self, current_position):
        """
        Updates the PID controller with the current position and calculates the new movement.

        Args:
            current_position (np.array): Current position of the robot (3D vector).

        Returns:
            np.array: The movement (3D vector) calculated by the PID controller.
        """
        # Calculate the error between the target and current positions
        error = self.target_position - current_position
        self.integral += error
        derivative = error - self.prev_error

        # PID formula for each axis
        movement = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Store current error for next update
        self.prev_error = error

        return movement  # Return the calculated movement to be applied to the robot


# Test a range of PID gains
pid_gains = [
    (0.5, 0.05, 0.05),
    (1.0, 0.1, 0.1),
    (1.5, 0.2, 0.1),
    (2.0, 0.1, 0.2),
    (0.8, 0.05, 0.15),
    (0.9, 0.1, 0.3),
    (1.2, 0.15, 0.05),
]

# Target position for testing
target_position = np.array([0.0, 0.0, 0.0])

# Store the best results
best_error = float('inf')
best_gains = None

for gains in pid_gains:
    kp, ki, kd = gains
    print(f"Testing PID gains: kp = {kp}, ki = {ki}, kd = {kd}")
    
    # Initialize the PID controller with the current set of gains
    controller = PIDController(kp, ki, kd, target_position)
    
    # Simulate movement towards the target position
    current_position = np.array([0.555, 0.555, 0.555])  # Starting position
    for step in range(100):  # Test for 100 steps
        # Update the PID controller and get the movement
        movement = controller.update_position(current_position)
        
        # Simulate the robot moving by applying the calculated movement
        current_position += movement
        
        # Calculate the error (distance to target position)
        error = np.linalg.norm(target_position - current_position)  # Euclidean distance
        
        # Check if the target position is reached
        if error < 0.01:  # Threshold for goal proximity
            print(f"Target position reached with gains: {kp}, {ki}, {kd}")
            break
    
    # Evaluate the final performance (error)
    final_error = np.linalg.norm(target_position - current_position)
    print(f"Final error with gains {kp}, {ki}, {kd}: {final_error:.4f} meters\n")
    
    # Track the best gains and performance
    if final_error < best_error:
        best_error = final_error
        best_gains = gains

print(f"Best PID Gains: kp = {best_gains[0]}, ki = {best_gains[1]}, kd = {best_gains[2]}")
print(f"Best Performance (Final Error): {best_error:.4f} meters")
