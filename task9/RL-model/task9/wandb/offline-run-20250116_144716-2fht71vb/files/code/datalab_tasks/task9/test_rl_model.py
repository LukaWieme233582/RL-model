import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from ot2_gym_wrapper import OT2GymWrapper  # Ensure this points to the correct wrapper

# Initialize W&B
config = {
    "policy_type": "MlpPolicy",  # Using Multi-Layer Perceptron for the policy
    "total_timesteps": 1000,  # Train for 1 million timesteps
    "learning_rate": 1e-4,
    "batch_size": 128,
    "gamma": 0.99,  # Default discount factor, will be modified
    "n_steps": 4096,
    "network_architecture": [128, 128],  # Two hidden layers of 128 neurons each
    "env_name": "OT2Env",  # Custom environment from Task 10
    "evaluation_episodes": 10
}

# Initialize W&B run
run = wandb.init(
    project="ot2-rl-training",
    config=config,
    sync_tensorboard=True,  # Auto-upload tensorboard metrics
    monitor_gym=True,  # Track video of agent performance
    save_code=True  # Optionally save the code in the W&B run
)

# Define function to create the environment
def make_env():
    env = OT2GymWrapper(render=False)  # Turn off rendering for training
    env = Monitor(env)  # Record stats like rewards and episodes
    return env

env = DummyVecEnv([make_env])  # Stable Baselines 3 expects a vectorized environment
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 1000 == 0,  # Record every 1000 steps
    video_length=200,
)

# Experiment list for different γ values
gamma_values = [0.96, 0.99, 0.999]

# Loop over each gamma value
for gamma_value in gamma_values:
    # Update config with current γ value
    config["gamma"] = gamma_value

    # Log the current gamma value for the experiment
    print(f"Running experiment with γ = {gamma_value}")

    # Update the model with the new gamma value
    model = PPO(config["policy_type"], env, gamma=config["gamma"], verbose=1, tensorboard_log=f"runs/{run.id}")

    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,  # Save gradients every 100 steps
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    # After training, evaluate the model
    print(f"Evaluating the model with γ = {gamma_value}")
    for episode in range(config["evaluation_episodes"]):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

# Finish the W&B run after all experiments
run.finish()

# Final Evaluation after all training
print("Final Evaluation:")
for episode in range(config["evaluation_episodes"]):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

