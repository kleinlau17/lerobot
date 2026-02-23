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
Record emotion gestures for SO101 robot arm to LeRobotDataset v3.0 (no video).

Two modes:
- leader: Record via SO101 Leader arm teleoperation (same flow as lerobot-record, video=False)
- programmatic: Write episodes from pre-defined keyframes without robot connection

Usage:

  # Leader recording (requires Leader + Follower arms connected):
  python record_emotion.py --mode=leader \\
      --robot.port=/dev/ttyACM0 \\
      --teleop.port=/dev/ttyACM1 \\
      --dataset.repo_id=user/emotion_gestures \\
      --dataset.single_task=happy

  # Programmatic (no hardware):
  python record_emotion.py --mode=programmatic \\
      --dataset.repo_id=user/emotion_gestures \\
      --dataset.root=./data/emotion_gestures
"""

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import log_say

FPS = 30

# Joint names for SO101 (order matters for feature names)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Pre-defined keyframes for programmatic mode (angles in degrees, gripper 0-100)
# Each emotion: list of (t_ratio, joints_dict) where t_ratio in [0, 1]
DEFAULT_KEYFRAMES: dict[str, list[tuple[float, dict[str, float]]]] = {
    "neutral": [
        (0.0, {j: 0.0 for j in JOINT_NAMES[:-1]} | {"gripper": 50.0}),
        (1.0, {j: 0.0 for j in JOINT_NAMES[:-1]} | {"gripper": 50.0}),
    ],
    "happy": [
        (0.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 10, "wrist_roll": 0, "gripper": 60}),
        (0.25, {"shoulder_pan": 5, "shoulder_lift": 22, "elbow_flex": -35, "wrist_flex": 12, "wrist_roll": 2, "gripper": 70}),
        (0.5, {"shoulder_pan": -5, "shoulder_lift": 18, "elbow_flex": -32, "wrist_flex": 8, "wrist_roll": -2, "gripper": 65}),
        (0.75, {"shoulder_pan": 5, "shoulder_lift": 22, "elbow_flex": -35, "wrist_flex": 12, "wrist_roll": 2, "gripper": 70}),
        (1.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 10, "wrist_roll": 0, "gripper": 60}),
    ],
    "sad": [
        (0.0, {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 40}),
        (0.5, {"shoulder_pan": 0, "shoulder_lift": -20, "elbow_flex": 20, "wrist_flex": -15, "wrist_roll": 0, "gripper": 30}),
        (1.0, {"shoulder_pan": 0, "shoulder_lift": -25, "elbow_flex": 25, "wrist_flex": -20, "wrist_roll": 0, "gripper": 25}),
    ],
    "curious": [
        (0.0, {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}),
        (0.5, {"shoulder_pan": 10, "shoulder_lift": 25, "elbow_flex": -40, "wrist_flex": 15, "wrist_roll": 5, "gripper": 55}),
        (1.0, {"shoulder_pan": 10, "shoulder_lift": 25, "elbow_flex": -40, "wrist_flex": 15, "wrist_roll": 5, "gripper": 55}),
    ],
    "wave": [
        (0.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}),
        (0.25, {"shoulder_pan": 25, "shoulder_lift": 20, "elbow_flex": -35, "wrist_flex": 5, "wrist_roll": 0, "gripper": 50}),
        (0.5, {"shoulder_pan": -25, "shoulder_lift": 20, "elbow_flex": -35, "wrist_flex": 5, "wrist_roll": 0, "gripper": 50}),
        (0.75, {"shoulder_pan": 25, "shoulder_lift": 20, "elbow_flex": -35, "wrist_flex": 5, "wrist_roll": 0, "gripper": 50}),
        (1.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}),
    ],
}


def _interpolate_keyframes(
    keyframes: list[tuple[float, dict[str, float]]], num_frames: int
) -> list[dict[str, float]]:
    """Linear interpolation between keyframes. Returns list of joint dicts."""
    import numpy as np

    t_ratios = np.array([k[0] for k in keyframes])
    traj = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        idx = np.searchsorted(t_ratios, t, side="right") - 1
        idx = max(0, min(idx, len(keyframes) - 2))
        t0, j0 = keyframes[idx]
        t1, j1 = keyframes[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        joints = {name: (1 - alpha) * j0[name] + alpha * j1[name] for name in JOINT_NAMES}
        traj.append(joints)
    return traj


def _joints_to_action_dict(joints: dict[str, float]) -> dict[str, float]:
    """Convert joints dict to action format {name.pos: val}."""
    return {f"{name}.pos": joints[name] for name in JOINT_NAMES}


def record_leader(
    repo_id: str,
    robot_port: str,
    teleop_port: str,
    single_task: str,
    episode_time_s: float = 30.0,
    reset_time_s: float = 10.0,
    num_episodes: int = 1,
    root: Path | None = None,
    push_to_hub: bool = False,
) -> LeRobotDataset:
    """Record emotion via Leader arm teleoperation (no video)."""
    robot_config = SO101FollowerConfig(
        port=robot_port,
        id="emotion_follower",
        cameras={},
        use_degrees=True,
    )
    teleop_config = SO101LeaderConfig(port=teleop_port, id="emotion_leader")

    robot = SO101Follower(robot_config)
    teleop = SO101Leader(teleop_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=False,
        ),
    )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=False,
    )

    robot.connect()
    teleop.connect()
    listener, events = init_keyboard_listener()

    try:
        if not robot.is_connected or not teleop.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        episode_idx = 0
        while episode_idx < num_episodes and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1}/{num_episodes} (task={single_task})")

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                dataset=dataset,
                control_time_s=episode_time_s,
                single_task=single_task,
                display_data=False,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if events["rerecord_episode"]:
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            if not events["stop_recording"] and episode_idx < num_episodes - 1:
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    control_time_s=reset_time_s,
                    single_task=single_task,
                    display_data=False,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            dataset.save_episode()
            episode_idx += 1

    finally:
        log_say("Stop recording")
        robot.disconnect()
        teleop.disconnect()
        listener.stop()
        dataset.finalize()
        if push_to_hub:
            dataset.push_to_hub()

    return dataset


def record_programmatic(
    repo_id: str,
    emotions: list[str] | None = None,
    duration_per_emotion_s: float = 2.0,
    root: Path | None = None,
    push_to_hub: bool = False,
    keyframes: dict[str, list[tuple[float, dict[str, float]]]] | None = None,
) -> LeRobotDataset:
    """Write emotion episodes from keyframes (no robot connection)."""
    kf = keyframes or DEFAULT_KEYFRAMES
    emotions = emotions or list(kf.keys())

    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="emotion_follower",
        cameras={},
        use_degrees=True,
    )
    robot = SO101Follower(robot_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=False,
        ),
    )

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=False,
    )

    for emotion_id in emotions:
        if emotion_id not in kf:
            logging.warning("Emotion '%s' not in keyframes, skipping", emotion_id)
            continue
        frames = int(duration_per_emotion_s * FPS)
        trajectory = _interpolate_keyframes(kf[emotion_id], frames)
        for t, joints in enumerate(trajectory):
            action_dict = _joints_to_action_dict(joints)
            obs_frame = build_dataset_frame(dataset.features, action_dict, prefix=OBS_STR)
            act_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
            frame = {**obs_frame, **act_frame, "task": emotion_id}
            dataset.add_frame(frame)
        dataset.save_episode()
        logging.info("Saved episode for emotion '%s' (%d frames)", emotion_id, frames)

    dataset.finalize()
    if push_to_hub:
        dataset.push_to_hub()
    return dataset


def main():
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Record emotion gestures for SO101")
    parser.add_argument("--mode", choices=["leader", "programmatic"], default="programmatic")
    parser.add_argument("--dataset.repo_id", dest="repo_id", default="user/emotion_gestures")
    parser.add_argument("--dataset.root", dest="root", type=str, default=None)
    parser.add_argument("--dataset.single_task", dest="single_task", default="neutral")
    parser.add_argument("--dataset.episode_time_s", dest="episode_time_s", type=float, default=15.0)
    parser.add_argument("--dataset.num_episodes", dest="num_episodes", type=int, default=1)
    parser.add_argument("--dataset.push_to_hub", dest="push_to_hub", action="store_true")
    parser.add_argument("--robot.port", dest="robot_port", default="/dev/ttyACM0")
    parser.add_argument("--teleop.port", dest="teleop_port", default="/dev/ttyACM1")
    parser.add_argument("--emotions", type=str, nargs="+", default=None)
    parser.add_argument("--duration_per_emotion_s", type=float, default=2.0)
    args = parser.parse_args()

    root = Path(args.root) if args.root else None

    if args.mode == "leader":
        record_leader(
            repo_id=args.repo_id,
            robot_port=args.robot_port,
            teleop_port=args.teleop_port,
            single_task=args.single_task,
            episode_time_s=args.episode_time_s,
            num_episodes=args.num_episodes,
            root=root,
            push_to_hub=args.push_to_hub,
        )
    else:
        record_programmatic(
            repo_id=args.repo_id,
            emotions=args.emotions,
            duration_per_emotion_s=args.duration_per_emotion_s,
            root=root,
            push_to_hub=args.push_to_hub,
        )


if __name__ == "__main__":
    main()
