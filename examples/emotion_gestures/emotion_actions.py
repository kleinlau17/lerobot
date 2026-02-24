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
EmotionActionKeyframes = list[tuple[float, dict[str, float]]]

EMOTION_ACTIONS: dict[str, dict[str, EmotionActionKeyframes]] = {
    "neutral": {
        "idle": [
            (0, {"elbow_flex": 24.20746684427722, "gripper": 3.8999999999999995, "shoulder_lift": -51.00002185735966, "shoulder_pan": 0, "wrist_flex": 25.649956848454543, "wrist_roll": 0.22915561607816304}),
            (1.5, {"elbow_flex": 35.05241199051345, "gripper": 12.7, "shoulder_lift": -62.000026571692146, "shoulder_pan": 5.059994261775338, "wrist_flex": 23.369960684147475, "wrist_roll": 0.22915561607816304}),
            (3, {"elbow_flex": 24.20746684427722, "gripper": 3.8999999999999995, "shoulder_lift": -51.00002185735966, "shoulder_pan": 0, "wrist_flex": 25.649956848454543, "wrist_roll": 0.22915561607816304}),
        ],
    },
    "happy": {
        "bounce": [
            (0, {"elbow_flex": -30, "gripper": 60, "shoulder_lift": 15, "shoulder_pan": 0, "wrist_flex": 10, "wrist_roll": 0}),
            (0.25, {"elbow_flex": -35, "gripper": 70, "shoulder_lift": 22, "shoulder_pan": 5, "wrist_flex": 12, "wrist_roll": 2}),
            (0.5, {"elbow_flex": -32, "gripper": 65, "shoulder_lift": 18, "shoulder_pan": -5, "wrist_flex": 8, "wrist_roll": -2}),
            (0.75, {"elbow_flex": -35, "gripper": 70, "shoulder_lift": 22, "shoulder_pan": 5, "wrist_flex": 12, "wrist_roll": 2}),
            (1, {"elbow_flex": -30, "gripper": 60, "shoulder_lift": 15, "shoulder_pan": 0, "wrist_flex": 10, "wrist_roll": 0}),
        ],
    },
    "sad": {
        "droop": [
            (0, {"elbow_flex": 0, "gripper": 40, "shoulder_lift": 0, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
            (0.5, {"elbow_flex": 20, "gripper": 30, "shoulder_lift": -20, "shoulder_pan": 0, "wrist_flex": -15, "wrist_roll": 0}),
            (1, {"elbow_flex": 25, "gripper": 25, "shoulder_lift": -25, "shoulder_pan": 0, "wrist_flex": -20, "wrist_roll": 0}),
        ],
    },
    "curious": {
        "lean": [
            (0, {"elbow_flex": 0, "gripper": 50, "shoulder_lift": 0, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
            (0.5, {"elbow_flex": -40, "gripper": 55, "shoulder_lift": 25, "shoulder_pan": 10, "wrist_flex": 15, "wrist_roll": 5}),
            (1, {"elbow_flex": -40, "gripper": 55, "shoulder_lift": 25, "shoulder_pan": 10, "wrist_flex": 15, "wrist_roll": 5}),
        ],
    },
    "wave": {
        "wave": [
            (0, {"elbow_flex": -30, "gripper": 50, "shoulder_lift": 15, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
            (0.25, {"elbow_flex": -35, "gripper": 50, "shoulder_lift": 20, "shoulder_pan": 25, "wrist_flex": 5, "wrist_roll": 0}),
            (0.5, {"elbow_flex": -35, "gripper": 50, "shoulder_lift": 20, "shoulder_pan": -25, "wrist_flex": 5, "wrist_roll": 0}),
            (0.75, {"elbow_flex": -35, "gripper": 50, "shoulder_lift": 20, "shoulder_pan": 25, "wrist_flex": 5, "wrist_roll": 0}),
            (1, {"elbow_flex": -30, "gripper": 50, "shoulder_lift": 15, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
        ],
    },
}
