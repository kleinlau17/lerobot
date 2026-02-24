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
Import emotion_actions from JSON (exported by robot_viewer KeyframePlugin) back
into Python structure compatible with emotion_actions.py.

Usage:
  # Write to emotion_actions.py (replaces EMOTION_ACTIONS and JOINT_NAMES):
  python import_emotion_json.py --input emotion_actions.json

  # Dry run: print new Python block to stdout:
  python import_emotion_json.py --input emotion_actions.json --dry-run

  # Write to a different file for review:
  python import_emotion_json.py --input emotion_actions.json --output emotion_actions_new.py
"""

import argparse
import json
from pathlib import Path


def format_joints(joints: dict) -> str:
    """Format joints dict as Python literal (ints for round numbers)."""
    parts = []
    for k, v in sorted(joints.items()):
        if isinstance(v, float) and v == int(v):
            parts.append(f'"{k}": {int(v)}')
        else:
            parts.append(f'"{k}": {v}')
    return "{" + ", ".join(parts) + "}"


def json_to_emotion_actions(data: dict, include_header: bool = False) -> str:
    """Generate EMOTION_ACTIONS and JOINT_NAMES Python source from JSON."""
    joint_names = data.get("joint_names", [])
    emotions = data.get("emotions", {})

    lines = []
    if include_header:
        lines.extend([
            "# -----------------------------------------------------------------------------",
            "# Joint names",
            "# -----------------------------------------------------------------------------",
            "",
        ])
    lines.extend([
        "JOINT_NAMES = [",
        *[f'    "{j}",' for j in joint_names],
        "]",
        "",
        "# -----------------------------------------------------------------------------",
        "# Action keyframes",
        "# -----------------------------------------------------------------------------",
        "",
        "# Pre-defined actions per emotion (angles in degrees, gripper 0-100).",
        "# Structure: emotion -> action_name -> list of (t_ratio in [0,1], joints_dict).",
        "EmotionActionKeyframes = list[tuple[float, dict[str, float]]]",
        "",
        "EMOTION_ACTIONS: dict[str, dict[str, EmotionActionKeyframes]] = {",
    ])
    for emotion, actions in emotions.items():
        lines.append(f'    "{emotion}": {{')
        for action_name, keyframes in actions.items():
            lines.append(f'        "{action_name}": [')
            for kf in keyframes:
                t = kf.get("t", 0)
                joints = kf.get("joints", {})
                lines.append(f"            ({t}, {format_joints(joints)}),")
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import emotion_actions from JSON into emotion_actions.py"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSON path")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output Python file (default: emotion_actions.py in same dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated EMOTION_ACTIONS block to stdout only",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    script_dir = Path(__file__).parent
    target = args.output or (script_dir / "emotion_actions.py")

    if args.dry_run:
        print(json_to_emotion_actions(data, include_header=True))
        return

    block = json_to_emotion_actions(data, include_header=True)

    if target == script_dir / "emotion_actions.py":
        # Replace in place: from "# --- # Joint names" to end of EMOTION_ACTIONS
        start_marker = "# -----------------------------------------------------------------------------\n# Joint names\n# -----------------------------------------------------------------------------"
        text = target.read_text(encoding="utf-8")
        i = text.find(start_marker)
        if i == -1:
            print("Could not find Joint names section in emotion_actions.py")
            return
        # Find end of EMOTION_ACTIONS: the closing "}" of the outer dict
        rest = text[i:]
        depth = 0
        end_pos = -1
        for j, c in enumerate(rest):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end_pos = j + 1
                    break
        if end_pos == -1:
            print("Could not find end of EMOTION_ACTIONS")
            return
        new_text = text[:i] + block + "\n" + text[i + end_pos :].lstrip()
        target.write_text(new_text, encoding="utf-8")
        print(f"Updated {target}")
    else:
        # Write full file
        header = '''#!/usr/bin/env python
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

'''
        target.write_text(header + block + "\n", encoding="utf-8")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
