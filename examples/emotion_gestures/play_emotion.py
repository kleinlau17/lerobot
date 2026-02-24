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
Play emotion gestures from LeRobotDataset on SO101 robot.

Supports replay by episode index or by emotion_id (task name). Reuses lerobot-replay logic.

Usage:

  # Replay by episode index (same as lerobot-replay):
  python play_emotion.py \\
      --robot.port=/dev/ttyACM0 \\
      --dataset.repo_id=user/emotion_gestures \\
      --dataset.episode=0

  # Replay by emotion_id (looks up episode with matching task):
  python play_emotion.py \\
      --robot.port=/dev/ttyACM0 \\
      --dataset.repo_id=user/emotion_gestures \\
      --dataset.emotion_id=happy

  # Use same robot.id as when recording (to reuse calibration file):
  python play_emotion.py --robot.id=black --robot.no_calibrate ...

  # Skip calibration (robot already calibrated, e.g. just used in another script):
  python play_emotion.py --robot.no_calibrate ...
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep


def find_episode_by_emotion(dataset: LeRobotDataset, emotion_id: str) -> int | None:
    """Find first episode index whose task matches emotion_id."""
    for ep_idx in range(dataset.num_episodes):
        ep_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
        if len(ep_frames) == 0:
            continue
        task_idx = ep_frames[0]["task_index"]
        task_idx_val = task_idx.item() if hasattr(task_idx, "item") else int(task_idx)
        task_val = dataset.meta.tasks.iloc[task_idx_val].name
        if task_val == emotion_id:
            return ep_idx
    return None


def replay_episode(
    robot: SO101Follower,
    dataset: LeRobotDataset,
    episode: int,
    num_repeats: int = 1,
    post_replay_delay: float = 2.0,
    play_sounds: bool = False,
    calibrate: bool = True,
) -> None:
    """Replay a single episode on the robot (same logic as lerobot-replay)."""
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode)
    actions = episode_frames.select_columns(ACTION)

    robot_action_processor = make_default_robot_action_processor()
    robot.connect(calibrate=calibrate)

    try:
        last_processed_action = None
        for repeat_idx in range(num_repeats):
            if repeat_idx > 0:
                logging.info("Repeat %d/%d", repeat_idx + 1, num_repeats)

            for idx in range(len(episode_frames)):
                start_t = time.perf_counter()
                action_array = actions[idx][ACTION]
                action = {dataset.features[ACTION]["names"][i]: action_array[i] for i in range(len(action_array))}
                robot_obs = robot.get_observation()
                processed = robot_action_processor((action, robot_obs))
                robot.send_action(processed)
                last_processed_action = processed
                dt = time.perf_counter() - start_t
                precise_sleep(max(1 / dataset.fps - dt, 0.0))

        if post_replay_delay > 0 and last_processed_action is not None:
            logging.info("Holding last action for %.1fs", post_replay_delay)
            num_hold = int(post_replay_delay * dataset.fps)
            for _ in range(num_hold):
                robot.send_action(last_processed_action)
                precise_sleep(1.0 / dataset.fps)
        elif post_replay_delay > 0:
            time.sleep(post_replay_delay)
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Play emotion gestures on SO101")
    parser.add_argument("--robot.port", dest="robot_port", default="/dev/ttyACM0")
    parser.add_argument("--robot.id", dest="robot_id", default="emotion_follower",
                        help="Must match the robot id used when recording to reuse calibration file")
    parser.add_argument("--robot.no_calibrate", dest="no_calibrate", action="store_true",
                        help="Skip calibration on connect (use when robot is already calibrated)")
    parser.add_argument("--dataset.repo_id", dest="repo_id", default="user/emotion_gestures")
    parser.add_argument("--dataset.root", dest="root", type=str, default=None)
    parser.add_argument("--dataset.episode", dest="episode", type=int, default=None)
    parser.add_argument("--dataset.emotion_id", dest="emotion_id", type=str, default=None)
    parser.add_argument("--dataset.num_repeats", dest="num_repeats", type=int, default=1)
    parser.add_argument("--dataset.post_replay_delay", dest="post_replay_delay", type=float, default=2.0)
    args = parser.parse_args()

    if args.episode is None and args.emotion_id is None:
        parser.error("Provide either --dataset.episode or --dataset.emotion_id")

    root = Path(args.root) if args.root else None
    dataset = LeRobotDataset(args.repo_id, root=root)

    if args.episode is not None:
        episode = args.episode
    else:
        episode = find_episode_by_emotion(dataset, args.emotion_id)
        if episode is None:
            raise ValueError(f"No episode found for emotion_id='{args.emotion_id}'")
        logging.info("Found episode %d for emotion '%s'", episode, args.emotion_id)

    robot_config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={},
        use_degrees=True,
    )
    robot = SO101Follower(robot_config)

    replay_episode(
        robot=robot,
        dataset=dataset,
        episode=episode,
        num_repeats=args.num_repeats,
        post_replay_delay=args.post_replay_delay,
        calibrate=not args.no_calibrate,
    )


if __name__ == "__main__":
    main()
