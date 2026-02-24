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
# Structure: emotion -> action_name -> list of (t in seconds, joints_dict).
# Animation duration = max(t) over keyframes; playback runs at that many seconds.
EmotionActionKeyframes = list[tuple[float, dict[str, float]]]

EMOTION_ACTIONS: dict[str, dict[str, EmotionActionKeyframes]] = {
    "neutral": {
        # Idle: 呼吸感微动，增加关键帧并做缓入缓出，避免顿挫
        "idle": [
            (0.0, {"elbow_flex": 24.2, "gripper": 3.9, "shoulder_lift": -51.0, "shoulder_pan": 0, "wrist_flex": 25.6, "wrist_roll": 0.2}),
            (0.4, {"elbow_flex": 26.4, "gripper": 5.4, "shoulder_lift": -53.2, "shoulder_pan": 1.0, "wrist_flex": 25.1, "wrist_roll": 0.2}),
            (1.1, {"elbow_flex": 32.2, "gripper": 10.2, "shoulder_lift": -58.5, "shoulder_pan": 3.8, "wrist_flex": 23.8, "wrist_roll": 0.2}),
            (1.5, {"elbow_flex": 35.0, "gripper": 12.7, "shoulder_lift": -62.0, "shoulder_pan": 5.0, "wrist_flex": 23.3, "wrist_roll": 0.2}),
            (1.9, {"elbow_flex": 32.2, "gripper": 10.2, "shoulder_lift": -58.5, "shoulder_pan": 3.8, "wrist_flex": 23.8, "wrist_roll": 0.2}),
            (2.6, {"elbow_flex": 26.4, "gripper": 5.4, "shoulder_lift": -53.2, "shoulder_pan": 1.0, "wrist_flex": 25.1, "wrist_roll": 0.2}),
            (3.0, {"elbow_flex": 24.2, "gripper": 3.9, "shoulder_lift": -51.0, "shoulder_pan": 0, "wrist_flex": 25.6, "wrist_roll": 0.2}),
        ],
    },
    "happy": {
        "bounce": [
            # t=0: 从默认的 idle 状态无缝开始
            (0, {"elbow_flex": 24.2, "gripper": 3.9, "shoulder_lift": -51.0, "shoulder_pan": 0, "wrist_flex": 25.6, "wrist_roll": 0.2}),
            
            # t=0.2: 快速下蹲蓄力 (大臂下沉，肘部回弯)
            (0.2, {"elbow_flex": 15.0, "gripper": 5.0, "shoulder_lift": -40.0, "shoulder_pan": -5.0, "wrist_flex": 35.0, "wrist_roll": -5.0}),
            
            # t=0.4: 向上弹起爆发 (大臂抬起，肘部伸展，嘴巴张开)
            (0.4, {"elbow_flex": 50.0, "gripper": 80.0, "shoulder_lift": -75.0, "shoulder_pan": 5.0, "wrist_flex": 5.0, "wrist_roll": 10.0}),
            
            # t=0.6: 在高点快速歪头抖动
            (0.6, {"elbow_flex": 45.0, "gripper": 40.0, "shoulder_lift": -70.0, "shoulder_pan": -5.0, "wrist_flex": 10.0, "wrist_roll": -10.0}),
            
            # t=0.8: 第二次弹起 (欢呼)
            (0.8, {"elbow_flex": 50.0, "gripper": 80.0, "shoulder_lift": -75.0, "shoulder_pan": 5.0, "wrist_flex": 5.0, "wrist_roll": 10.0}),
            
            # t=1.2: 平稳回落到默认状态
            (1.2, {"elbow_flex": 24.2, "gripper": 3.9, "shoulder_lift": -51.0, "shoulder_pan": 0, "wrist_flex": 25.6, "wrist_roll": 0.2}),
        ],
    },
    "sad": {
        # "droop": [
        #     (0, {"elbow_flex": 0, "gripper": 40, "shoulder_lift": 0, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
        #     (0.5, {"elbow_flex": 20, "gripper": 30, "shoulder_lift": -20, "shoulder_pan": 0, "wrist_flex": -15, "wrist_roll": 0}),
        #     (1, {"elbow_flex": 25, "gripper": 25, "shoulder_lift": -25, "shoulder_pan": 0, "wrist_flex": -20, "wrist_roll": 0}),
        # ],
        "slump": [
            # 0s: 从 idle 开始
            (0.0, {"shoulder_pan": 0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.2, "gripper": 3.9}),
            # 2.0s: 缓慢垂下头，身体微向前倾但由于肘部收缩，重心依然安全
            (2.0, {"shoulder_pan": 0, "shoulder_lift": -40.0, "elbow_flex": 15.0, "wrist_flex": 65.0, "wrist_roll": 5.0, "gripper": 0.0}),
            # 4.0s: 像叹气一样，夹爪微张后闭合
            (4.0, {"shoulder_pan": 0, "shoulder_lift": -42.0, "elbow_flex": 18.0, "wrist_flex": 60.0, "wrist_roll": 5.0, "gripper": 10.0}),
            (5.0, {"shoulder_pan": 0, "shoulder_lift": -40.0, "elbow_flex": 15.0, "wrist_flex": 65.0, "wrist_roll": 5.0, "gripper": 0.0}),
        ],
    },
    "curious": {
        # "lean": [
        #     (0, {"elbow_flex": 0, "gripper": 50, "shoulder_lift": 0, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
        #     (0.5, {"elbow_flex": -40, "gripper": 55, "shoulder_lift": 25, "shoulder_pan": 10, "wrist_flex": 15, "wrist_roll": 5}),
        #     (1, {"elbow_flex": -40, "gripper": 55, "shoulder_lift": 25, "shoulder_pan": 10, "wrist_flex": 15, "wrist_roll": 5}),
        # ],
        "tilt": [
            (0.0, {"shoulder_pan": 0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.2, "gripper": 10.0}),
            # 1.0s: 向左偏移并歪头
            (1.0, {"shoulder_pan": 15.0, "shoulder_lift": -55.0, "elbow_flex": 28.0, "wrist_flex": 30.0, "wrist_roll": 25.0, "gripper": 10.0}),
            # 2.5s: 换个角度，向右偏移并反向歪头
            (2.5, {"shoulder_pan": -15.0, "shoulder_lift": -55.0, "elbow_flex": 28.0, "wrist_flex": 30.0, "wrist_roll": -25.0, "gripper": 10.0}),
            (4.0, {"shoulder_pan": 0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.2, "gripper": 10.0}),
        ]
    },
    "wave": {
        # "wave": [
        #     (0, {"elbow_flex": -30, "gripper": 50, "shoulder_lift": 15, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
        #     (0.25, {"elbow_flex": -35, "gripper": 50, "shoulder_lift": 20, "shoulder_pan": 25, "wrist_flex": 5, "wrist_roll": 0}),
        #     (0.5, {"elbow_flex": -35, "gripper": 50, "shoulder_lift": 20, "shoulder_pan": -25, "wrist_flex": 5, "wrist_roll": 0}),
        #     (0.75, {"elbow_flex": -35, "gripper": 50, "shoulder_lift": 20, "shoulder_pan": 25, "wrist_flex": 5, "wrist_roll": 0}),
        #     (1, {"elbow_flex": -30, "gripper": 50, "shoulder_lift": 15, "shoulder_pan": 0, "wrist_flex": 0, "wrist_roll": 0}),
        # ],
        "shiver": [
            # 0s: idle
            (0.0, {"shoulder_pan": 0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.2, "gripper": 3.9}),
            # 0.1s - 0.5s: 快速小幅左右震颤 (Pan轴)
            (0.1, {"shoulder_pan": 5.0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.0, "gripper": 80.0}),
            (0.2, {"shoulder_pan": -5.0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.0, "gripper": 5.0}),
            (0.3, {"shoulder_pan": 5.0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.0, "gripper": 80.0}),
            (0.4, {"shoulder_pan": -5.0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.0, "gripper": 5.0}),
            # 最后猛地向前探一下（小幅）
            (0.7, {"shoulder_pan": 0, "shoulder_lift": -45.0, "elbow_flex": 20.0, "wrist_flex": 20.0, "wrist_roll": 0.0, "gripper": 5.0}),
            (1.5, {"shoulder_pan": 0, "shoulder_lift": -51.0, "elbow_flex": 24.2, "wrist_flex": 25.6, "wrist_roll": 0.2, "gripper": 3.9}),
        ]
    },
}
