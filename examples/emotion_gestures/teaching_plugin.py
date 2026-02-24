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
Teaching plugin for emotion gesture keyframes.

Provides a small Python helper class to:
- Manage emotions / actions / keyframes in-memory
- Record keyframes from a callback that returns current joints
- Import / export JSON compatible with export_emotion_json.py / robot_viewer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple
import json

from emotion_actions import EMOTION_ACTIONS as DEFAULT_EMOTION_ACTIONS, JOINT_NAMES as DEFAULT_JOINT_NAMES


EmotionActionKeyframes = List[Tuple[float, Dict[str, float]]]


def _deepcopy_emotions(
    emotions: MutableMapping[str, MutableMapping[str, Iterable[Tuple[float, Dict[str, float]]]]]
) -> Dict[str, Dict[str, EmotionActionKeyframes]]:
    copied: Dict[str, Dict[str, EmotionActionKeyframes]] = {}
    for emotion, actions in emotions.items():
        copied[emotion] = {}
        for action_name, keyframes in actions.items():
            kf_list: EmotionActionKeyframes = []
            for t, joints in keyframes:
                kf_list.append((float(t), {str(k): float(v) for k, v in joints.items()}))
            copied[emotion][action_name] = kf_list
    return copied


def _auto_time_values(n: int) -> List[float]:
    if n <= 0:
        return []
    if n == 1:
        return [0.0]
    step = 1.0 / (n - 1)
    return [i * step for i in range(n)]


@dataclass
class EmotionTeachingConfig:
    joint_names: List[str]
    gripper_range: Tuple[float, float] = (0.0, 100.0)


class EmotionTeachingSession:
    """
    In-memory teaching session for emotion keyframes.

    Data layout:
        emotions: dict[emotion][action] -> list[(t_ratio, joints_dict)]
    """

    def __init__(
        self,
        get_current_joints: Optional[Callable[[], Dict[str, float]]] = None,
        *,
        emotions: Optional[MutableMapping[str, MutableMapping[str, Iterable[Tuple[float, Dict[str, float]]]]]] = None,
        joint_names: Optional[Iterable[str]] = None,
        gripper_range: Tuple[float, float] = (0.0, 100.0),
    ) -> None:
        if emotions is None:
            emotions = DEFAULT_EMOTION_ACTIONS
            if joint_names is None:
                joint_names = DEFAULT_JOINT_NAMES
        if joint_names is None:
            joint_names = DEFAULT_JOINT_NAMES

        self._config = EmotionTeachingConfig(joint_names=list(joint_names), gripper_range=tuple(gripper_range))
        self._emotions: Dict[str, Dict[str, EmotionActionKeyframes]] = _deepcopy_emotions(emotions)
        self.get_current_joints = get_current_joints

    # ------------------------------------------------------------------
    # Configuration accessors
    # ------------------------------------------------------------------
    @property
    def joint_names(self) -> List[str]:
        return list(self._config.joint_names)

    @property
    def gripper_range(self) -> Tuple[float, float]:
        return self._config.gripper_range

    # ------------------------------------------------------------------
    # Emotion / action listing helpers
    # ------------------------------------------------------------------
    def get_emotions(self) -> List[str]:
        return sorted(self._emotions.keys())

    def get_actions(self, emotion: str) -> List[str]:
        if emotion not in self._emotions:
            return []
        return sorted(self._emotions[emotion].keys())

    def get_keyframes(self, emotion: str, action: str) -> EmotionActionKeyframes:
        return list(self._emotions.get(emotion, {}).get(action, []))

    # ------------------------------------------------------------------
    # Emotion / action management
    # ------------------------------------------------------------------
    def add_emotion(self, name: str) -> None:
        if name not in self._emotions:
            self._emotions[name] = {}

    def remove_emotion(self, name: str) -> None:
        self._emotions.pop(name, None)

    def rename_emotion(self, old: str, new: str) -> None:
        if old == new or old not in self._emotions:
            return
        if new in self._emotions:
            raise ValueError(f"Emotion '{new}' already exists")
        self._emotions[new] = self._emotions.pop(old)

    def add_action(self, emotion: str, action: str) -> None:
        if emotion not in self._emotions:
            self._emotions[emotion] = {}
        if action not in self._emotions[emotion]:
            self._emotions[emotion][action] = []

    def remove_action(self, emotion: str, action: str) -> None:
        if emotion in self._emotions:
            self._emotions[emotion].pop(action, None)

    def rename_action(self, emotion: str, old: str, new: str) -> None:
        if emotion not in self._emotions or old not in self._emotions[emotion]:
            return
        if old == new:
            return
        if new in self._emotions[emotion]:
            raise ValueError(f"Action '{new}' already exists in emotion '{emotion}'")
        self._emotions[emotion][new] = self._emotions[emotion].pop(old)

    # ------------------------------------------------------------------
    # Keyframe operations
    # ------------------------------------------------------------------
    def record_keyframe(self, emotion: str, action: str, t: Optional[float] = None) -> None:
        """
        Capture current joints from callback and add as a keyframe.

        If t is None, keyframe times for this action are re-spaced uniformly in [0, 1].
        """
        if self.get_current_joints is None:
            raise RuntimeError("get_current_joints callback is not set")

        joints_raw = self.get_current_joints()
        joints: Dict[str, float] = {}
        for name in self._config.joint_names:
            v = joints_raw.get(name, 50.0 if name == "gripper" else 0.0)
            joints[name] = float(v)

        if emotion not in self._emotions:
            self._emotions[emotion] = {}
        if action not in self._emotions[emotion]:
            self._emotions[emotion][action] = []

        keyframes = self._emotions[emotion][action]
        if t is None:
            all_joints = [kf[1] for kf in keyframes] + [joints]
            times = _auto_time_values(len(all_joints))
            self._emotions[emotion][action] = list(zip(times, all_joints))
        else:
            keyframes.append((float(t), joints))

    def set_keyframe(
        self,
        emotion: str,
        action: str,
        index: int,
        *,
        t: Optional[float] = None,
        joints: Optional[Dict[str, float]] = None,
    ) -> None:
        keyframes = self._emotions.get(emotion, {}).get(action)
        if not keyframes:
            raise IndexError("No keyframes for given emotion/action")
        if index < 0 or index >= len(keyframes):
            raise IndexError("Keyframe index out of range")

        old_t, old_joints = keyframes[index]
        new_t = float(t) if t is not None else old_t
        if joints is not None:
            new_joints: Dict[str, float] = {}
            for name in self._config.joint_names:
                v = joints.get(name, 50.0 if name == "gripper" else 0.0)
                new_joints[name] = float(v)
        else:
            new_joints = dict(old_joints)
        keyframes[index] = (new_t, new_joints)

    def delete_keyframe(self, emotion: str, action: str, index: int) -> None:
        keyframes = self._emotions.get(emotion, {}).get(action)
        if not keyframes:
            return
        if 0 <= index < len(keyframes):
            del keyframes[index]

    # ------------------------------------------------------------------
    # Data export helpers
    # ------------------------------------------------------------------
    def get_data(self) -> Dict[str, Dict[str, EmotionActionKeyframes]]:
        return _deepcopy_emotions(self._emotions)

    def to_json_dict(self) -> Dict:
        emotions_data: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        for emotion, actions in self._emotions.items():
            emotions_data[emotion] = {}
            for action_name, keyframes in actions.items():
                emotions_data[emotion][action_name] = [
                    {"t": float(t), "joints": dict(joints)} for t, joints in keyframes
                ]

        return {
            "joint_names": list(self._config.joint_names),
            "gripper_range": [float(self._config.gripper_range[0]), float(self._config.gripper_range[1])],
            "emotions": emotions_data,
        }

    # ------------------------------------------------------------------
    # JSON / dict constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_json(
        cls,
        json_path: Path | str,
        get_current_joints: Optional[Callable[[], Dict[str, float]]] = None,
    ) -> "EmotionTeachingSession":
        path = Path(json_path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        joint_names = data.get("joint_names", list(DEFAULT_JOINT_NAMES))
        emotions = data.get("emotions", {})
        gripper_range_list = data.get("gripper_range", [0.0, 100.0])
        if len(gripper_range_list) != 2:
            gripper_range = (0.0, 100.0)
        else:
            gripper_range = (float(gripper_range_list[0]), float(gripper_range_list[1]))

        # emotions: emotion -> action_name -> [{"t", "joints"}, ...]
        parsed_emotions: Dict[str, Dict[str, EmotionActionKeyframes]] = {}
        for emotion, actions in emotions.items():
            parsed_emotions[emotion] = {}
            for action_name, keyframes in actions.items():
                kf_list: EmotionActionKeyframes = []
                for kf in keyframes:
                    t = float(kf.get("t", 0.0))
                    joints_dict = {
                        str(k): float(v) for k, v in kf.get("joints", {}).items()
                    }
                    kf_list.append((t, joints_dict))
                parsed_emotions[emotion][action_name] = kf_list

        return cls(
            get_current_joints=get_current_joints,
            emotions=parsed_emotions,
            joint_names=joint_names,
            gripper_range=gripper_range,
        )

    @classmethod
    def from_emotions(
        cls,
        emotions: MutableMapping[str, MutableMapping[str, Iterable[Tuple[float, Dict[str, float]]]]],
        *,
        joint_names: Optional[Iterable[str]] = None,
        gripper_range: Tuple[float, float] = (0.0, 100.0),
        get_current_joints: Optional[Callable[[], Dict[str, float]]] = None,
    ) -> "EmotionTeachingSession":
        return cls(
            get_current_joints=get_current_joints,
            emotions=emotions,
            joint_names=joint_names,
            gripper_range=gripper_range,
        )

    @classmethod
    def from_python_defaults(
        cls,
        get_current_joints: Optional[Callable[[], Dict[str, float]]] = None,
    ) -> "EmotionTeachingSession":
        return cls(get_current_joints=get_current_joints)


def flatten_emotions_to_default_keyframes(
    emotions: MutableMapping[str, MutableMapping[str, Iterable[Tuple[float, Dict[str, float]]]]]
) -> Dict[str, List[Tuple[float, Dict[str, float]]]]:
    """
    Flatten emotion->action->keyframes into emotion->keyframes by picking one action per emotion.

    For each emotion, prefers an action named:
        emotion, "idle", "default", "main"
    falling back to the lexicographically first action.
    """
    result: Dict[str, List[Tuple[float, Dict[str, float]]]] = {}
    for emotion, actions in emotions.items():
        if not actions:
            continue
        chosen_name: Optional[str] = None
        for candidate in (emotion, "idle", "default", "main"):
            if candidate in actions:
                chosen_name = candidate
                break
        if chosen_name is None:
            chosen_name = sorted(actions.keys())[0]
        kf_list_raw = list(actions[chosen_name])
        kf_list: List[Tuple[float, Dict[str, float]]] = []
        for t, joints in kf_list_raw:
            kf_list.append((float(t), {str(k): float(v) for k, v in joints.items()}))
        result[emotion] = kf_list
    return result


def flatten_json_to_default_keyframes(json_path: Path | str) -> Dict[str, List[Tuple[float, Dict[str, float]]]]:
    """
    Convenience helper for record_emotion.py:
    Load a JSON file in export_emotion_json.py format and flatten it into
    DEFAULT_KEYFRAMES-style mapping: emotion -> list[(t, joints)].
    """
    session = EmotionTeachingSession.from_json(json_path)
    return flatten_emotions_to_default_keyframes(session.get_data())

