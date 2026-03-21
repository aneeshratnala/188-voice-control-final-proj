"""
Voice-driven NutAssembly test — mirrors ``test.py`` (stack) flow.

- Say **start** to open the sim, then **assemble** / **grab** / jog words.
- **assemble**: full waypoint run (square nut then round nut) without hover/place prompts.
- **grab**: segmented grab → say **hover** → say **place** for each nut; after the square nut,
  say **grab** or **assemble** to continue (same choices as between stack layers).
"""
import numpy as np
import robosuite as suite
from policies import *

from voice_start import (
    KEYWORDS_INCREMENTAL,
    IncrementalMicMonitor,
    StopMicMonitor,
    wait_for_assemble_grab_or_incremental,
    wait_for_hover,
    wait_for_place,
    wait_for_start,
)

# Robosuite control rate only (no wall-clock sleep — runs at full sim speed like CA1_PID_starter).
CONTROL_FREQ = 20

MAX_STEPS_PHASE = 12000
POST_PLACE_OPEN_STEPS = 30


def _action_open_gripper_hold() -> np.ndarray:
    a = np.zeros(7)
    a[-1] = -1.0
    return a


def _env_success(env, reward: float, done: bool) -> bool:
    if reward >= 0.99 or done:
        return True
    check = getattr(env, "_check_success", None)
    if callable(check):
        try:
            return bool(check())
        except Exception:
            return False
    return False


# 1) Voice: open sim only after "start"
vosk_model = wait_for_start()
stop_monitor = StopMicMonitor(vosk_model)
stop_monitor.start()
stop_monitor.pause()

# 2) NutAssembly (stock robosuite task)
env = suite.make(
    env_name="NutAssembly",
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
    env.render()
    obs_holder[0], _, _, _ = env.step(_action_open_gripper_hold())


# 3) assemble | grab | incremental jog
command = wait_for_assemble_grab_or_incremental(
    vosk_model, between_blocks=_idle_while_listening
)

if command in KEYWORDS_INCREMENTAL:
    inc_monitor = IncrementalMicMonitor(vosk_model)
    inc_monitor.start()
    inc_monitor.resume()
    policy = IncrementalTeleopPolicy(obs_holder[0])
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
    finally:
        inc_monitor.close()
else:
    stop_monitor.resume()

    def _listen_with_policy(policy: NutAssemblyPolicy) -> None:
        o = obs_holder[0]
        obs_holder[0], _, _, _ = env.step(policy.get_action(o))
        env.render()

    def _run_place_until(
        policy: NutAssemblyPolicy,
        *,
        until_round_pause: bool,
    ) -> None:
        """Step through place/release until square→round handoff (-2) or full task success."""
        steps = 0
        while True:
            stop_monitor.check()
            action = policy.get_action(obs_holder[0])
            obs_holder[0], reward, done, _ = env.step(action)
            env.render()
            steps += 1
            if until_round_pause:
                if policy.current_nut == "Round" and policy.waypoint == -2:
                    break
            elif _env_success(env, reward, done):
                break
            if steps >= MAX_STEPS_PHASE:
                print(
                    "Phase timed out — check placement / contacts; stopping phase.",
                    flush=True,
                )
                break

    def _run_segmented_one_nut(policy: NutAssemblyPolicy, *, until_round_pause: bool) -> None:
        listen = lambda: _listen_with_policy(policy)

        while not policy.segment_grab_done(obs_holder[0]):
            stop_monitor.check()
            obs_holder[0], _, _, _ = env.step(policy.get_action(obs_holder[0]))
            env.render()

        stop_monitor.pause()
        wait_for_hover(vosk_model, between_blocks=listen)
        stop_monitor.resume()
        policy.begin_hover()

        while not policy.segment_hover_done(obs_holder[0]):
            stop_monitor.check()
            obs_holder[0], _, _, _ = env.step(policy.get_action(obs_holder[0]))
            env.render()

        stop_monitor.pause()
        wait_for_place(vosk_model, between_blocks=listen)
        stop_monitor.resume()
        policy.begin_place()

        _run_place_until(policy, until_round_pause=until_round_pause)

        for _ in range(POST_PLACE_OPEN_STEPS):
            stop_monitor.check()
            obs_holder[0], _, _, _ = env.step(_action_open_gripper_hold())
            env.render()

    def _run_full_assemble(policy: NutAssemblyPolicy) -> None:
        steps = 0
        while True:
            stop_monitor.check()
            action = policy.get_action(obs_holder[0])
            obs_holder[0], reward, done, _ = env.step(action)
            env.render()
            steps += 1
            if _env_success(env, reward, done):
                print("Nut assembly success (env reward / done).", flush=True)
                break
            if steps >= MAX_STEPS_PHASE * 2:
                print("Full assemble timed out.", flush=True)
                break

    def _begin_round_nut_full_auto(policy: NutAssemblyPolicy) -> None:
        """After square nut in segmented mode: user said **assemble** — finish round nut without voice."""
        policy.segmented = False
        eef = obs_holder[0]["robot0_eef_pos"]
        policy.retreat_start_pos = eef[:3].copy()
        policy.waypoint = -1

    segmented = command == "grab"

    if not segmented:
        policy = NutAssemblyPolicy(obs_holder[0], segmented=False)
        print('Heard "assemble" — running full nut assembly (both nuts).', flush=True)
        _run_full_assemble(policy)
    else:
        print(
            'Heard "grab" — segmented square nut (say "hover" then "place").',
            flush=True,
        )
        policy = NutAssemblyPolicy(obs_holder[0], segmented=True)
        _run_segmented_one_nut(policy, until_round_pause=True)

        if not (policy.current_nut == "Round" and policy.waypoint == -2):
            print("Square-nut phase did not reach round-nut pause; exiting.", flush=True)
        else:
            print(
                'Round nut: say "grab" (segmented) or "assemble" (continuous), or a jog command.',
                flush=True,
            )
            layer_cmd = wait_for_assemble_grab_or_incremental(
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
                finally:
                    inc_monitor.close()
            elif layer_cmd == "assemble":
                _begin_round_nut_full_auto(policy)
                print('Heard "assemble" — running round nut automatically.', flush=True)
                _run_full_assemble(policy)
            else:
                # layer_cmd == "grab" (same pattern as stack: keyword starts the next layer)
                policy.start_next_grab_segment(obs_holder[0])
                print('Heard "grab" — segmented round nut.', flush=True)
                _run_segmented_one_nut(policy, until_round_pause=False)
                if _env_success(env, 0.0, False):
                    print("Nut assembly success (env).", flush=True)
                print("Finished round nut segment.", flush=True)
