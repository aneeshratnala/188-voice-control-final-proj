import numpy as np
import robosuite as suite
from policies import *

from voice_start import wait_for_move, wait_for_start

# 1) Voice: open sim only after "start"
vosk_model = wait_for_start()

# 2) One episode instance — same reset for idle listening and stacking (no new episode on "move")
env = suite.make(
    env_name="Stack",  # replace with "NutAssembly" and "Door" (1)
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=5000,
)

obs_holder = [env.reset()]


def _idle_while_listening() -> None:
    """Keep the window responsive; do not advance the manipulation policy."""
    env.render()
    zero = np.zeros(7)
    obs_holder[0], _, _, _ = env.step(zero)


# 3) Voice: begin stacking only after "move" (still the same episode)
wait_for_move(vosk_model, between_blocks=_idle_while_listening)

# 4) Complete stacking in this episode only — do not reset here
obs = obs_holder[0]
policy = StackPolicy(obs)  ## CHANGE NAME HERE (2)

while True:
    action = policy.get_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if reward == 1.0 or done:
        break
