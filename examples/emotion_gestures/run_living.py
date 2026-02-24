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
import json
import logging
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

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

# Queue for external emotion triggers (latest wins)
_emotion_trigger_queue: "queue.Queue[str]" = queue.Queue(maxsize=1)


class _TriggerRequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for /trigger endpoint to accept emotion triggers."""

    # Disable default logging to stderr
    def log_message(self, format: str, *args) -> None:  # type: ignore[override]
        return

    def do_POST(self) -> None:  # type: ignore[override]
        if self.path != "/trigger":
            self.send_response(404)
            self.end_headers()
            return

        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length or "0")
        except ValueError:
            length = 0

        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        emotion = data.get("emotion")
        if not isinstance(emotion, str):
            self.send_response(400)
            self.end_headers()
            return

        # Accept only known emotions; treat neutral as idle for the state machine
        if emotion not in {"idle", "happy", "sad", "curious", "wave", "neutral"}:
            self.send_response(400)
            self.end_headers()
            return

        normalized = "idle" if emotion == "neutral" else emotion

        try:
            # Keep only the latest trigger
            while True:
                try:
                    _emotion_trigger_queue.get_nowait()
                except queue.Empty:
                    break
            _emotion_trigger_queue.put_nowait(normalized)
        except queue.Full:
            # Should not happen due to clearing above, but ignore if it does
            pass

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')


def start_trigger_server(port: int) -> None:
    """Start background HTTP server for external emotion triggers."""

    server = HTTPServer(("127.0.0.1", port), _TriggerRequestHandler)

    def _serve() -> None:
        try:
            server.serve_forever()
        except Exception:
            # Best-effort server; shutdown on any unexpected error
            logging.exception("Trigger HTTP server stopped unexpectedly")

    thread = threading.Thread(target=_serve, name="emotion-trigger-server", daemon=True)
    thread.start()
    logging.info("Emotion trigger HTTP server listening on http://127.0.0.1:%d/trigger", port)


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

            # Apply any pending external emotion triggers
            try:
                while True:
                    trigger = _emotion_trigger_queue.get_nowait()
                    state_machine.trigger_emotion(trigger)
            except queue.Empty:
                pass

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
    parser.add_argument(
        "--trigger.port",
        dest="trigger_port",
        type=int,
        default=9998,
        help="Local HTTP port for external emotion triggers",
    )
    args = parser.parse_args()

    # Start background HTTP trigger server
    start_trigger_server(args.trigger_port)

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
