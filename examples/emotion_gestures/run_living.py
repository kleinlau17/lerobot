#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run living behavior on SO101 robot (Pixar Luxo Jr style).

State machine switches between idle (breathing + micro-motion) and emotions
(happy, sad, curious, wave). Does NOT depend on record_emotion or play_emotion.

Usage:

  python run_living.py --robot.port=/dev/ttyACM0

  # With options:
  python run_living.py --robot.port=/dev/ttyACM0 \\
      --breathing.amplitude=4 --expression.interval_min_s=20 \\
      --emotion.duration_s=2.5
"""

import argparse
import logging
import time

from lerobot.processor import make_default_robot_action_processor
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep

from emotion_actions import EMOTION_ACTIONS
from living_behavior import (
    EmotionStateMachine,
    LivingBehaviorGenerator,
    joints_to_action_dict,
)

FPS = 30


def run_living(
    robot: SO101Follower,
    *,
    fps: int = 30,
    breathing_amplitude: float = 3.0,
    breathing_freq_hz: float = 0.25,
    expression_interval_min_s: float = 30.0,
    expression_interval_max_s: float = 90.0,
    emotion_duration_s: float = 3.0,
    micro_fidget_amplitude: float = 2.0,
    calibrate: bool = True,
) -> None:
    """Run living behavior loop. Press Ctrl+C to stop."""
    state_machine = EmotionStateMachine(
        interval_min_s=expression_interval_min_s,
        interval_max_s=expression_interval_max_s,
        emotion_duration_s=emotion_duration_s,
    )
    generator = LivingBehaviorGenerator(
        breathing_amplitude=breathing_amplitude,
        breathing_freq_hz=breathing_freq_hz,
        micro_fidget_amplitude=micro_fidget_amplitude,
    )
    robot_action_processor = make_default_robot_action_processor()

    robot.connect(calibrate=calibrate)
    frame_interval = 1.0 / fps
    t0 = time.perf_counter()
    last_state: str | None = None

    try:
        while True:
            start_t = time.perf_counter()
            t = start_t - t0

            state, state_local_t = state_machine.step(t)
            if state != last_state:
                if state == "idle":
                    print("情绪: idle | 动作: breathing + micro-fidget", flush=True)
                else:
                    action_name = next(iter(EMOTION_ACTIONS[state]))
                    print(f"情绪: {state} | 动作: {action_name}", flush=True)
                last_state = state

            joints = generator.get_frame(state, t, state_local_t, fps=float(fps))
            action_dict = joints_to_action_dict(joints)
            robot_obs = robot.get_observation()
            processed = robot_action_processor((action_dict, robot_obs))
            robot.send_action(processed)

            dt = time.perf_counter() - start_t
            precise_sleep(max(frame_interval - dt, 0.0))
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run Pixar Luxo Jr style living behavior on SO101"
    )
    parser.add_argument("--robot.port", dest="robot_port", default="/dev/ttyACM0")
    parser.add_argument("--robot.id", dest="robot_id", default="my_awesome_follower_arm")
    parser.add_argument(
        "--robot.no_calibrate",
        dest="no_calibrate",
        action="store_true",
        help="Skip calibration (robot already calibrated)",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--breathing.amplitude", dest="breathing_amplitude", type=float, default=3.0)
    parser.add_argument("--breathing.freq_hz", dest="breathing_freq_hz", type=float, default=0.25)
    parser.add_argument(
        "--expression.interval_min_s",
        dest="expression_interval_min_s",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--expression.interval_max_s",
        dest="expression_interval_max_s",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--emotion.duration_s",
        dest="emotion_duration_s",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--micro_fidget.amplitude",
        dest="micro_fidget_amplitude",
        type=float,
        default=2.0,
    )
    args = parser.parse_args()

    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={},
        use_degrees=True,
    )
    robot = SO101Follower(config)

    run_living(
        robot,
        fps=args.fps,
        breathing_amplitude=args.breathing_amplitude,
        breathing_freq_hz=args.breathing_freq_hz,
        expression_interval_min_s=args.expression_interval_min_s,
        expression_interval_max_s=args.expression_interval_max_s,
        emotion_duration_s=args.emotion_duration_s,
        micro_fidget_amplitude=args.micro_fidget_amplitude,
        calibrate=not args.no_calibrate,
    )


if __name__ == "__main__":
    main()
