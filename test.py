import numpy as np
import robosuite as suite
from policies import *
import time

from voice_start import wait_for_start

wait_for_start()

# create environment instance
env = suite.make(
    env_name="Stack", # replace with other tasks "NutAssembly" and "Door" (1)
    robots="Panda",  
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=5000,  # increase episode length to allow robot to complete full task
)

# reset the environment
for _ in range(5):
    obs = env.reset()
    policy = StackPolicy(obs)   ## CHANGE NAME HERE (2)
    
    while True:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)  # take action in the environment
        
        env.render()  # render on display
        # time.sleep(0.05)
        if reward == 1.0 or done: break
