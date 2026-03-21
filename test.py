import numpy as np
import robosuite as suite
from policies import *

from voice_start import (
    wait_for_hover,
    wait_for_place,
    wait_for_stack_or_grab,
    wait_for_start,
)

# 1) Voice: open sim only after "start"
vosk_model = wait_for_start()

# 2) One episode instance
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
    """Idle before first command (no policy yet / not segmented)."""
    env.render()
    obs_holder[0], _, _, _ = env.step(np.zeros(7))


# 3) "stack" = full run | "grab" = grab → (voice) hover → (voice) place
command = wait_for_stack_or_grab(vosk_model, between_blocks=_idle_while_listening)

obs = obs_holder[0]
segmented = command == "grab"
policy = StackPolicy(obs, segmented=segmented)  ## CHANGE NAME HERE (2)


def _listen_with_policy() -> None:
    """While waiting for voice, keep sim stepping with current policy (hold poses)."""
    o = obs_holder[0]
    obs_holder[0], _, _, _ = env.step(policy.get_action(o))
    env.render()


def _run_until_stack_done() -> None:
    while True:
        o = obs_holder[0]
        action = policy.get_action(o)
        obs_holder[0], reward, done, info = env.step(action)
        env.render()
        if reward == 1.0 or done:
            break


if segmented:
    # Grab: phases 0–1, then hold (phase 10) — no B motion yet
    while not policy.segment_grab_done(obs_holder[0]):
        o = obs_holder[0]
        obs_holder[0], _, _, _ = env.step(policy.get_action(o))
        env.render()

    wait_for_hover(vosk_model, between_blocks=_listen_with_policy)
    policy.begin_hover()

    # Hover: existing phase-2 waypoint above green cube, then hold (phase 11)
    while not policy.segment_hover_done(obs_holder[0]):
        o = obs_holder[0]
        obs_holder[0], _, _, _ = env.step(policy.get_action(o))
        env.render()

    wait_for_place(vosk_model, between_blocks=_listen_with_policy)
    policy.begin_place()

# Place (segmented) or full stack: existing phase-3 place-down + release
_run_until_stack_done()
