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
Interactive CLI for teaching emotion keyframes.

This tool lets you:
- load existing emotion keyframes from emotion_actions.py or JSON
- create / select emotions and actions
- record keyframes by manually entering joint values
- adjust / delete keyframes
- export back to JSON compatible with robot_viewer KeyframePlugin

Usage examples:

  # From Python defaults (emotion_actions.EMOTION_ACTIONS)
  python cli_teach_emotions.py

  # From JSON exported by export_emotion_json.py or robot_viewer
  python cli_teach_emotions.py --from-json emotion_actions.json

  # Specify output path
  python cli_teach_emotions.py --output-json my_emotion_actions.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from teaching_plugin import EmotionTeachingSession


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw:
            if default is not None:
                return default
            print("请输入一个数字。")
            continue
        try:
            return float(raw)
        except ValueError:
            print("格式不正确，请重新输入数字。")


def _prompt_int(prompt: str, min_value: int, max_value: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("请输入整数。")
            continue
        if value < min_value or value > max_value:
            print(f"请输入 [{min_value}, {max_value}] 范围内的整数。")
            continue
        return value


def _prompt_choice(prompt: str, choices: List[str]) -> str:
    while True:
        name = input(prompt).strip()
        if name in choices:
            return name
        print(f"无效名称，可选项：{', '.join(choices) or '(空)'}")


def _make_manual_get_current_joints(joint_names: List[str]):
    def _get_current_joints() -> Dict[str, float]:
        print("\n请输入当前姿态的关节值（单位：度，gripper 0-100）。")
        print("直接回车可使用默认值（非 gripper 为 0，gripper 为 50）。")
        joints: Dict[str, float] = {}
        for name in joint_names:
            default = 50.0 if name == "gripper" else 0.0
            value = _prompt_float(f"{name} [{default}]: ", default=default)
            joints[name] = value
        return joints

    return _get_current_joints


def interactive_loop(session: EmotionTeachingSession, output_path: Path) -> None:
    current_emotion: str | None = None
    current_action: str | None = None

    while True:
        print("\n================ Emotion Teaching =================")
        print(f"关节: {', '.join(session.joint_names)}")
        print(f"当前 emotion: {current_emotion or '-'} | action: {current_action or '-'}")
        print("---------------------------------------------------")
        print("1) 列出所有 emotions")
        print("2) 选择 emotion")
        print("3) 新建 emotion")
        print("4) 重命名 emotion")
        print("5) 删除 emotion")
        print("6) 列出当前 emotion 的 actions")
        print("7) 选择 action")
        print("8) 新建 action")
        print("9) 删除 action")
        print("10) 列出当前 action 的 keyframes")
        print("11) 录制一个 keyframe（从当前姿态）")
        print("12) 修改 keyframe 的时间 t ∈ [0,1]")
        print("13) 删除 keyframe")
        print("14) 导出为 JSON 并退出")
        print("0) 退出（不保存）")

        choice = input("请选择操作编号: ").strip()

        if choice == "1":
            emotions = session.get_emotions()
            if not emotions:
                print("暂无 emotions。")
            else:
                print("Emotions:")
                for e in emotions:
                    print(f"- {e}")

        elif choice == "2":
            emotions = session.get_emotions()
            if not emotions:
                print("暂无 emotion，请先新建。")
                continue
            name = _prompt_choice("输入要选择的 emotion 名称: ", emotions)
            current_emotion = name
            actions = session.get_actions(current_emotion)
            current_action = actions[0] if actions else None

        elif choice == "3":
            name = input("输入新 emotion 名称: ").strip()
            if not name:
                print("名称不能为空。")
                continue
            session.add_emotion(name)
            current_emotion = name
            current_action = None
            print(f"已创建 emotion: {name}")

        elif choice == "4":
            emotions = session.get_emotions()
            if not emotions:
                print("暂无 emotion。")
                continue
            old = _prompt_choice("输入要重命名的 emotion: ", emotions)
            new = input("输入新的 emotion 名称: ").strip()
            if not new:
                print("新名称不能为空。")
                continue
            try:
                session.rename_emotion(old, new)
            except ValueError as e:
                print(f"重命名失败: {e}")
                continue
            if current_emotion == old:
                current_emotion = new
            print(f"已将 emotion '{old}' 重命名为 '{new}'")

        elif choice == "5":
            emotions = session.get_emotions()
            if not emotions:
                print("暂无 emotion。")
                continue
            name = _prompt_choice("输入要删除的 emotion: ", emotions)
            session.remove_emotion(name)
            if current_emotion == name:
                current_emotion = None
                current_action = None
            print(f"已删除 emotion: {name}")

        elif choice == "6":
            if not current_emotion:
                print("请先选择 emotion（操作 2 或 3）。")
                continue
            actions = session.get_actions(current_emotion)
            if not actions:
                print(f"emotion '{current_emotion}' 下暂无 actions。")
            else:
                print(f"Actions in '{current_emotion}':")
                for a in actions:
                    print(f"- {a}")

        elif choice == "7":
            if not current_emotion:
                print("请先选择 emotion（操作 2 或 3）。")
                continue
            actions = session.get_actions(current_emotion)
            if not actions:
                print(f"emotion '{current_emotion}' 下暂无 actions，请先新建。")
                continue
            name = _prompt_choice("输入要选择的 action 名称: ", actions)
            current_action = name

        elif choice == "8":
            if not current_emotion:
                print("请先选择 emotion（操作 2 或 3）。")
                continue
            name = input("输入新 action 名称: ").strip()
            if not name:
                print("名称不能为空。")
                continue
            session.add_action(current_emotion, name)
            current_action = name
            print(f"已在 emotion '{current_emotion}' 下创建 action: {name}")

        elif choice == "9":
            if not current_emotion:
                print("请先选择 emotion（操作 2 或 3）。")
                continue
            actions = session.get_actions(current_emotion)
            if not actions:
                print(f"emotion '{current_emotion}' 下暂无 actions。")
                continue
            name = _prompt_choice("输入要删除的 action 名称: ", actions)
            session.remove_action(current_emotion, name)
            if current_action == name:
                current_action = None
            print(f"已删除 action: {name}")

        elif choice == "10":
            if not current_emotion or not current_action:
                print("请先选择 emotion 和 action（操作 2/3 + 7/8）。")
                continue
            keyframes = session.get_keyframes(current_emotion, current_action)
            if not keyframes:
                print(f"emotion '{current_emotion}', action '{current_action}' 下暂无 keyframes。")
                continue
            print(f"Keyframes for {current_emotion} / {current_action}:")
            for idx, (t, joints) in enumerate(keyframes):
                summary = ", ".join(
                    f"{name}={joints.get(name, 0):.1f}"
                    for name in session.joint_names
                    if name in joints
                )
                print(f"[{idx}] t={t:.3f} | {summary}")

        elif choice == "11":
            if not current_emotion or not current_action:
                print("请先选择 emotion 和 action（操作 2/3 + 7/8）。")
                continue
            session.record_keyframe(current_emotion, current_action, t=None)
            keyframes = session.get_keyframes(current_emotion, current_action)
            idx = len(keyframes) - 1
            t, joints = keyframes[idx]
            print(f"已添加 keyframe index={idx}, t={t:.3f}")
            summary = ", ".join(
                f"{name}={joints.get(name, 0):.1f}"
                for name in session.joint_names
                if name in joints
            )
            print(f"姿态: {summary}")

        elif choice == "12":
            if not current_emotion or not current_action:
                print("请先选择 emotion 和 action（操作 2/3 + 7/8）。")
                continue
            keyframes = session.get_keyframes(current_emotion, current_action)
            if not keyframes:
                print("暂无 keyframes。")
                continue
            print(f"当前共有 {len(keyframes)} 个 keyframes。")
            idx = _prompt_int("请输入要修改的 keyframe index: ", 0, len(keyframes) - 1)
            new_t = _prompt_float("输入新的 t 值 (0-1): ")
            if not (0.0 <= new_t <= 1.0):
                print("t 必须在 [0, 1] 范围内。")
                continue
            session.set_keyframe(current_emotion, current_action, idx, t=new_t)
            print(f"已更新 keyframe [{idx}] 的 t={new_t:.3f}")

        elif choice == "13":
            if not current_emotion or not current_action:
                print("请先选择 emotion 和 action（操作 2/3 + 7/8）。")
                continue
            keyframes = session.get_keyframes(current_emotion, current_action)
            if not keyframes:
                print("暂无 keyframes。")
                continue
            print(f"当前共有 {len(keyframes)} 个 keyframes。")
            idx = _prompt_int("请输入要删除的 keyframe index: ", 0, len(keyframes) - 1)
            session.delete_keyframe(current_emotion, current_action, idx)
            print(f"已删除 keyframe [{idx}]")

        elif choice == "14":
            data = session.to_json_dict()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                import json

                json.dump(data, f, indent=2)
            print(f"\n已导出到 JSON: {output_path}")
            print("可以使用 `import_emotion_json.py` 将其写回 emotion_actions.py，")
            print("或在 robot_viewer 中载入该 JSON 进行可视化。")
            break

        elif choice == "0":
            print("未保存任何修改，直接退出。")
            break

        else:
            print("无效的选择，请重新输入。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive teaching tool for emotion keyframes.")
    parser.add_argument(
        "--mode",
        choices=["manual", "robot"],
        default="manual",
        help="选择示教模式: manual=命令行手动输入关节值; robot=从 SO101 从臂当前姿态读取关节值。",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="从 JSON 文件初始化（与 export_emotion_json.py 格式兼容）。若不指定则从 emotion_actions.EMOTION_ACTIONS 初始化。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).parent / "emotion_actions_taught.json",
        help="导出 JSON 的路径（默认: emotion_actions_taught.json）。",
    )
    parser.add_argument(
        "--robot.port",
        dest="robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="当 mode=robot 时，SO101 从臂串口端口（默认: /dev/ttyACM0）。",
    )
    parser.add_argument(
        "--robot.id",
        dest="robot_id",
        type=str,
        default="emotion_follower_teach",
        help="当 mode=robot 时，从臂在日志中的标识符（默认: emotion_follower_teach）。",
    )
    args = parser.parse_args()

    # 初始化示教会话（从 Python 默认或 JSON）
    if args.from_json is not None:
        session = EmotionTeachingSession.from_json(args.from_json)
    else:
        session = EmotionTeachingSession.from_python_defaults()

    if args.mode == "manual":
        # 使用命令行手动输入的方式获取当前关节姿态
        session.get_current_joints = _make_manual_get_current_joints(session.joint_names)
        interactive_loop(session, args.output_json)
        return

    # mode == "robot": 连接 SO101 从臂，从真实关节姿态读取 keyframe
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={},
        use_degrees=True,
    )
    robot = SO101Follower(config)

    def _get_current_joints_from_robot() -> Dict[str, float]:
        obs = robot.get_observation()
        joints: Dict[str, float] = {}
        for name in session.joint_names:
            key = f"{name}.pos"
            if key in obs:
                joints[name] = float(obs[key])
        return joints

    session.get_current_joints = _get_current_joints_from_robot

    try:
        robot.connect()
        interactive_loop(session, args.output_json)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

