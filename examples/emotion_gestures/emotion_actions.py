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
Emotion action keyframes for SO101 robot arm (Pixar Luxo Jr style).

Decoupled action definitions: JOINT_NAMES, EMOTION_ACTIONS.
Used by living_behavior and can be shared with record_emotion.
"""

# -----------------------------------------------------------------------------
# Joint names
# -----------------------------------------------------------------------------

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# -----------------------------------------------------------------------------
# Action keyframes
# -----------------------------------------------------------------------------

# Pre-defined actions per emotion (angles in degrees, gripper 0-100).
# Structure: emotion -> action_name -> list of (t_ratio in [0,1], joints_dict).
# Each action is the minimal unit: one keyframe sequence playable as a trajectory.
EmotionActionKeyframes = list[tuple[float, dict[str, float]]]

EMOTION_ACTIONS: dict[str, dict[str, EmotionActionKeyframes]] = {
    "neutral": {
        "idle": [
            (0.0, {j: 0.0 for j in JOINT_NAMES[:-1]} | {"gripper": 50.0}),
            (1.0, {j: 0.0 for j in JOINT_NAMES[:-1]} | {"gripper": 50.0}),
        ],
    },
    "happy": {
        "bounce": [
            (0.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 10, "wrist_roll": 0, "gripper": 60}),
            (0.25, {"shoulder_pan": 5, "shoulder_lift": 22, "elbow_flex": -35, "wrist_flex": 12, "wrist_roll": 2, "gripper": 70}),
            (0.5, {"shoulder_pan": -5, "shoulder_lift": 18, "elbow_flex": -32, "wrist_flex": 8, "wrist_roll": -2, "gripper": 65}),
            (0.75, {"shoulder_pan": 5, "shoulder_lift": 22, "elbow_flex": -35, "wrist_flex": 12, "wrist_roll": 2, "gripper": 70}),
            (1.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 10, "wrist_roll": 0, "gripper": 60}),
        ],
    },
    "sad": {
        "droop": [
            (0.0, {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 40}),
            (0.5, {"shoulder_pan": 0, "shoulder_lift": -20, "elbow_flex": 20, "wrist_flex": -15, "wrist_roll": 0, "gripper": 30}),
            (1.0, {"shoulder_pan": 0, "shoulder_lift": -25, "elbow_flex": 25, "wrist_flex": -20, "wrist_roll": 0, "gripper": 25}),
        ],
    },
    "curious": {
        "lean": [
            (0.0, {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}),
            (0.5, {"shoulder_pan": 10, "shoulder_lift": 25, "elbow_flex": -40, "wrist_flex": 15, "wrist_roll": 5, "gripper": 55}),
            (1.0, {"shoulder_pan": 10, "shoulder_lift": 25, "elbow_flex": -40, "wrist_flex": 15, "wrist_roll": 5, "gripper": 55}),
        ],
    },
    "wave": {
        "wave": [
            (0.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}),
            (0.25, {"shoulder_pan": 25, "shoulder_lift": 20, "elbow_flex": -35, "wrist_flex": 5, "wrist_roll": 0, "gripper": 50}),
            (0.5, {"shoulder_pan": -25, "shoulder_lift": 20, "elbow_flex": -35, "wrist_flex": 5, "wrist_roll": 0, "gripper": 50}),
            (0.75, {"shoulder_pan": 25, "shoulder_lift": 20, "elbow_flex": -35, "wrist_flex": 5, "wrist_roll": 0, "gripper": 50}),
            (1.0, {"shoulder_pan": 0, "shoulder_lift": 15, "elbow_flex": -30, "wrist_flex": 0, "wrist_roll": 0, "gripper": 50}),
        ],
    },
}
