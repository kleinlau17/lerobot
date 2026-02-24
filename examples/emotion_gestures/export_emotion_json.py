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
Export emotion_actions.py to JSON for robot_viewer KeyframePlugin.

Output format:
{
  "joint_names": [...],
  "gripper_range": [0, 100],
  "emotions": {
    "emotion_name": {
      "action_name": [{"t": 0.0, "joints": {...}}, ...]
    }
  }
}

Usage:
  python export_emotion_json.py [--output emotion_actions.json]
"""

import argparse
import json
from pathlib import Path

from emotion_actions import EMOTION_ACTIONS, JOINT_NAMES


def export_to_json(output_path: Path) -> None:
    emotions_data = {}
    for emotion, actions in EMOTION_ACTIONS.items():
        emotions_data[emotion] = {}
        for action_name, keyframes in actions.items():
            emotions_data[emotion][action_name] = [
                {"t": t, "joints": dict(joints)}
                for t, joints in keyframes
            ]

    data = {
        "joint_names": list(JOINT_NAMES),
        "gripper_range": [0, 100],
        "emotions": emotions_data,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Exported to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export emotion_actions to JSON for robot_viewer"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).parent / "emotion_actions.json",
        help="Output JSON path",
    )
    args = parser.parse_args()
    export_to_json(args.output)


if __name__ == "__main__":
    main()
