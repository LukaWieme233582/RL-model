import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, target_position):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_position = target_position
        self.prev_error = 0
        self.integral = 0
        self.current_position = np.array([0.555, 0.555, 0.555])  # Initial position
        
    def update_position(self):
        error = self.target_position - self.current_position
        self.integral += error
        derivative = error - self.prev_error
        
        # PID formula for each axis
        movement = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.current_position += movement  # Update the position
        
        self.prev_error = error
        
        return self.current_position

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
    for step in range(100):  # Test for 100 steps
        current_position = controller.update_position()
        error = np.linalg.norm(target_position - current_position)  # Euclidean distance
        if error < 0.01:
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
