import time

import numpy as np
import robosuite as suite
import stack_extra_env
from policies import *

from voice_start import (
    KEYWORDS_INCREMENTAL,
    IncrementalMicMonitor,
    StopMicMonitor,
    wait_for_hover,
    wait_for_place,
    wait_for_stack_grab_or_incremental,
    wait_for_start,
)

# Match robosuite default control rate; throttle wall clock so 1 sim step ≈ 1/control_freq seconds.
CONTROL_FREQ = 20
STEP_DT = 1.0 / CONTROL_FREQ

# Red on green, then each extra block on the current tower top (see stack_extra_env.py).
STACK_LAYERS = [
    ("cubeA", "cubeB"),
    ("decor_blue", "cubeA"),
    ("decor_light", "decor_blue"),
    ("decor_dark", "decor_light"),
    ("decor_gray", "decor_dark"),
]

MAX_STEPS_PER_LAYER = 8000
# Match StackPolicy / OSC: -1 = open; 0 leaves gripper ambiguous and can stay partly closed.
POST_PLACE_OPEN_STEPS = 30


def _action_open_gripper_hold() -> np.ndarray:
    a = np.zeros(7)
    a[-1] = -1.0
    return a


def patch_stack_obs(obs: dict, manip: str, target: str) -> dict:
    """Feed StackPolicy: grasp *manip*, place on *target* (names match object obs keys)."""
    out = dict(obs)
    out["cubeA_pos"] = np.asarray(obs[f"{manip}_pos"], dtype=np.float64, copy=True)
    out["cubeB_pos"] = np.asarray(obs[f"{target}_pos"], dtype=np.float64, copy=True)
    return out


def throttle_realtime() -> None:
    time.sleep(STEP_DT)


# 1) Voice: open sim only after "start"
vosk_model = wait_for_start()
stop_monitor = StopMicMonitor(vosk_model)
stop_monitor.start()
stop_monitor.pause()

# 2) One episode instance (extra cubes + tower stacking via patched obs)
env = suite.make(
    env_name="StackExtraCubes",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    control_freq=CONTROL_FREQ,
    horizon=5000,
    ignore_done=True,
)

obs_holder = [env.reset()]


def _idle_while_listening() -> None:
    """Idle while listening for voice; keep gripper open so the arm does not drag the last cube."""
    env.render()
    obs_holder[0], _, _, _ = env.step(_action_open_gripper_hold())
    throttle_realtime()


# 3) stack | grab | incremental jog (same grammar as original Stack demo)
command = wait_for_stack_grab_or_incremental(
    vosk_model, between_blocks=_idle_while_listening
)

obs = obs_holder[0]

if command in KEYWORDS_INCREMENTAL:
    # Background mic for jog + stop; leave stop_monitor paused (no double mic).
    inc_monitor = IncrementalMicMonitor(vosk_model)
    inc_monitor.start()
    inc_monitor.resume()
    policy = IncrementalTeleopPolicy(obs)
    policy.set_command(command)
    print(f'Heard "{command}" — incremental mode (say jog words or "stop").', flush=True)
    try:
        while True:
            inc_monitor.check_stop()
            for w in inc_monitor.drain_commands():
                print(f'Heard "{w}" — applying.', flush=True)
                policy.set_command(w)
            o = obs_holder[0]
            obs_holder[0], _, _, _ = env.step(policy.get_action(o))
            env.render()
            throttle_realtime()
    finally:
        inc_monitor.close()
else:
    stop_monitor.resume()

    def _listen_with_policy(policy, manip: str, target: str) -> None:
        o = obs_holder[0]
        po = patch_stack_obs(o, manip, target)
        obs_holder[0], _, _, _ = env.step(policy.get_action(po))
        env.render()
        throttle_realtime()

    def _run_place_until_done(
        policy, manip: str, target: str, *, first_layer: bool
    ) -> None:
        steps = 0
        while True:
            stop_monitor.check()
            po = patch_stack_obs(obs_holder[0], manip, target)
            action = policy.get_action(po)
            obs_holder[0], reward, done, _ = env.step(action)
            env.render()
            throttle_realtime()
            steps += 1
            if first_layer:
                if reward >= 0.99 or done or env._check_success():
                    break
            else:
                if env.check_upper_on_lower(manip, target):
                    break
            if steps >= MAX_STEPS_PER_LAYER:
                print(
                    "Layer timed out — check contacts / placement; continuing if more layers.",
                    flush=True,
                )
                break

    def _run_layer(manip: str, target: str, segmented: bool) -> None:
        policy = StackPolicy(
            patch_stack_obs(obs_holder[0], manip, target), segmented=segmented
        )

        listen = lambda: _listen_with_policy(policy, manip, target)

        if segmented:
            while not policy.segment_grab_done(
                patch_stack_obs(obs_holder[0], manip, target)
            ):
                stop_monitor.check()
                po = patch_stack_obs(obs_holder[0], manip, target)
                obs_holder[0], _, _, _ = env.step(policy.get_action(po))
                env.render()
                throttle_realtime()

            stop_monitor.pause()
            wait_for_hover(vosk_model, between_blocks=listen)
            stop_monitor.resume()
            policy.begin_hover()

            while not policy.segment_hover_done(
                patch_stack_obs(obs_holder[0], manip, target)
            ):
                stop_monitor.check()
                po = patch_stack_obs(obs_holder[0], manip, target)
                obs_holder[0], _, _, _ = env.step(policy.get_action(po))
                env.render()
                throttle_realtime()

            stop_monitor.pause()
            wait_for_place(vosk_model, between_blocks=listen)
            stop_monitor.resume()
            policy.begin_place()

        first = manip == "cubeA" and target == "cubeB"
        _run_place_until_done(policy, manip, target, first_layer=first)
        for _ in range(POST_PLACE_OPEN_STEPS):
            stop_monitor.check()
            obs_holder[0], _, _, _ = env.step(_action_open_gripper_hold())
            env.render()
            throttle_realtime()

    layer_cmd = command
    for layer_idx, (manip, target) in enumerate(STACK_LAYERS):
        if layer_idx > 0:
            print(
                f'Layer {layer_idx + 1}/{len(STACK_LAYERS)}: say "stack", "grab", or a jog command.',
                flush=True,
            )
            layer_cmd = wait_for_stack_grab_or_incremental(
                vosk_model, between_blocks=_idle_while_listening
            )
            if layer_cmd in KEYWORDS_INCREMENTAL:
                inc_monitor = IncrementalMicMonitor(vosk_model)
                inc_monitor.start()
                inc_monitor.resume()
                inc_policy = IncrementalTeleopPolicy(obs_holder[0])
                inc_policy.set_command(layer_cmd)
                print(
                    f'Heard "{layer_cmd}" — incremental mode (say jog words or "stop").',
                    flush=True,
                )
                try:
                    while True:
                        inc_monitor.check_stop()
                        for w in inc_monitor.drain_commands():
                            print(f'Heard "{w}" — applying.', flush=True)
                            inc_policy.set_command(w)
                            o = obs_holder[0]
                            obs_holder[0], _, _, _ = env.step(inc_policy.get_action(o))
                            env.render()
                            throttle_realtime()
                finally:
                    inc_monitor.close()
                break

        segmented = layer_cmd == "grab"
        _run_layer(manip, target, segmented=segmented)
        print(f"Finished layer {manip} → {target}.", flush=True)
