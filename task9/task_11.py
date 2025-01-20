from stable_baselines3 import PPO
import gymnasium as gym
import time
from ot2_gym_wrapper_2 import OT2Env
import argparse
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task



parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
args = parser.parse_args()


os.environ['WANDB_API_KEY']='b099c2c4fdb64b69c352b1d21fe26ea4314d363c'
run = wandb.init(project='Test1_gamma', sync_tensorboard=True)


env = OT2Env(render=False, max_steps=1000)

model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=0.0001,
            batch_size=128,
            gamma=args.gamma,
            n_steps= 4096,
            n_epochs=10,
            tensorboard_log=f'runs/{run.id}'
            )

wandb_callback = WandbCallback(
    model_save_freq = 10000,
    model_save_path = f'models/{run.id}',
    verbose = 2
)

# variable for how often to save the model
timesteps = 1000000
for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps*(i+1)}")

env.clos()
wandb.finish()

