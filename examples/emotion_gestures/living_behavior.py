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
Living behavior for SO101 robot arm (Pixar Luxo Jr style).

Uses emotion_actions for keyframe data. EmotionStateMachine and LivingBehaviorGenerator.
Does NOT import record_emotion or play_emotion.
"""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np

from emotion_actions import EMOTION_ACTIONS, JOINT_NAMES, EmotionActionKeyframes

# -----------------------------------------------------------------------------
# Emotion states (idle = breathing + micro-motion; others = keyframe playback)
EmotionState = Literal["idle", "happy", "sad", "curious", "wave"]

# Emotions that can be randomly chosen from idle (sad typically via external trigger)
RANDOM_EMOTIONS: list[EmotionState] = ["happy", "curious", "wave"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def get_keyframes(emotion: str, action: str | None = None) -> EmotionActionKeyframes:
    """Get keyframes for an emotion and optional action.
    If action is None and the emotion has only one action, returns that action."""
    actions = EMOTION_ACTIONS[emotion]
    if action is None:
        action = next(iter(actions))
    return actions[action]


def interpolate_keyframes(
    keyframes: list[tuple[float, dict[str, float]]],
    num_frames: int,
) -> list[dict[str, float]]:
    """Linear interpolation between keyframes. Returns list of joint dicts."""
    t_ratios = np.array([k[0] for k in keyframes])
    traj = []
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        idx = int(np.searchsorted(t_ratios, t, side="right") - 1)
        idx = max(0, min(idx, len(keyframes) - 2))
        t0, j0 = keyframes[idx]
        t1, j1 = keyframes[idx + 1]
        alpha = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        joints = {name: (1 - alpha) * j0[name] + alpha * j1[name] for name in JOINT_NAMES}
        traj.append(joints)
    return traj


def joints_to_action_dict(joints: dict[str, float]) -> dict[str, float]:
    """Convert joints dict to action format {name.pos: val}."""
    return {f"{name}.pos": joints[name] for name in JOINT_NAMES}


# -----------------------------------------------------------------------------
# State machine
# -----------------------------------------------------------------------------


class EmotionStateMachine:
    """
    State machine for switching between idle and emotion states.
    - idle: stays for interval_min_s..interval_max_s, then randomly picks emotion
    - emotion: plays for emotion_duration_s, then returns to idle
    - trigger_emotion(emotion): external trigger to switch immediately
    """

    def __init__(
        self,
        *,
        interval_min_s: float = 30.0,
        interval_max_s: float = 90.0,
        emotion_duration_s: float = 3.0,
        emotion_weights: dict[str, float] | None = None,
        include_sad_in_random: bool = False,
    ):
        self.interval_min_s = interval_min_s
        self.interval_max_s = interval_max_s
        self.emotion_duration_s = emotion_duration_s
        self.emotion_weights = emotion_weights or {
            "happy": 0.4,
            "curious": 0.3,
            "wave": 0.3,
        }
        if include_sad_in_random:
            self.emotion_weights["sad"] = 0.1
            # Renormalize
            total = sum(self.emotion_weights.values())
            self.emotion_weights = {k: v / total for k, v in self.emotion_weights.items()}

        self.state: EmotionState = "idle"
        self.state_entered_t: float = 0.0
        self.next_idle_interval_s: float = random.uniform(interval_min_s, interval_max_s)
        self._pending_trigger: str | None = None

    def trigger_emotion(self, emotion: str) -> None:
        """Request immediate switch to the given emotion on next step."""
        if emotion in ("idle", "happy", "sad", "curious", "wave"):
            self._pending_trigger = emotion

    def step(self, t: float) -> tuple[EmotionState, float]:
        """
        Advance state machine. Returns (current_state, state_local_t).
        state_local_t = time elapsed since entering current state.
        """
        state_local_t = t - self.state_entered_t

        # External trigger takes priority
        if self._pending_trigger is not None:
            em = self._pending_trigger
            self._pending_trigger = None
            self.state = em  # type: ignore
            self.state_entered_t = t
            return (self.state, 0.0)

        if self.state == "idle":
            # Check if we should transition to an emotion
            if state_local_t >= self.next_idle_interval_s:
                weights = [(e, self.emotion_weights.get(e, 0)) for e in RANDOM_EMOTIONS]
                weights = [(e, w) for e, w in weights if w > 0]
                if not weights:
                    weights = [("happy", 1.0)]
                choices, probs = zip(*weights)
                chosen = random.choices(choices, weights=probs, k=1)[0]
                self.state = chosen
                self.state_entered_t = t
                return (self.state, 0.0)
            return (self.state, state_local_t)

        else:
            # In an emotion state: check if animation finished
            if state_local_t >= self.emotion_duration_s:
                self.state = "idle"
                self.state_entered_t = t
                self.next_idle_interval_s = random.uniform(
                    self.interval_min_s, self.interval_max_s
                )
                return (self.state, 0.0)
            return (self.state, state_local_t)


# -----------------------------------------------------------------------------
# Behavior generator
# -----------------------------------------------------------------------------


class LivingBehaviorGenerator:
    """
    Generates joint positions for living behavior:
    - idle: base pose + breathing (sine) + micro-fidget (small sine/perturbation)
    - happy/sad/curious/wave: interpolated keyframe trajectory
    """

    def __init__(
        self,
        *,
        breathing_amplitude: float = 3.0,
        breathing_freq_hz: float = 0.25,
        micro_fidget_amplitude: float = 2.0,
        micro_fidget_freq_hz: float = 0.15,
    ):
        self.breathing_amplitude = breathing_amplitude
        self.breathing_freq_hz = breathing_freq_hz
        self.micro_fidget_amplitude = micro_fidget_amplitude
        self.micro_fidget_freq_hz = micro_fidget_freq_hz

        # Precompute emotion x action trajectories (at 30 fps, 4 s = 120 frames)
        self._emotion_action_trajectories: dict[str, dict[str, list[dict[str, float]]]] = {}
        for emo in ("happy", "sad", "curious", "wave"):
            self._emotion_action_trajectories[emo] = {}
            for action_name, kf in EMOTION_ACTIONS[emo].items():
                num_frames = 120  # 4 s at 30 fps
                self._emotion_action_trajectories[emo][action_name] = interpolate_keyframes(
                    kf, num_frames
                )

        # Base pose (neutral idle)
        neutral_idle_kf = get_keyframes("neutral", "idle")
        self._base_pose = neutral_idle_kf[0][1]

        # Fixed phases for micro-fidget (deterministic per joint for smooth motion)
        self._micro_phases = {j: random.uniform(0, 2 * math.pi) for j in JOINT_NAMES}

    def get_frame(
        self,
        state: EmotionState,
        t: float,
        state_local_t: float,
        fps: float = 30.0,
        action_name: str | None = None,
    ) -> dict[str, float]:
        """
        Get joint positions for current state at given times.
        Returns dict {joint_name: angle_deg}.
        If state is an emotion, action_name selects which action to play;
        if None, uses the first (default) action for that emotion.
        """
        if state == "idle":
            return self._get_idle_frame(t)
        else:
            actions = self._emotion_action_trajectories[state]
            if action_name is None:
                action_name = next(iter(actions))
            traj = actions[action_name]
            num_frames = len(traj)
            frame_idx = min(
                int(state_local_t * fps),
                num_frames - 1,
            )
            return traj[frame_idx].copy()

    def _get_idle_frame(self, t: float) -> dict[str, float]:
        """Idle: base + breathing + micro-fidget."""
        joints = dict(self._base_pose)

        # Breathing: shoulder_lift, elbow_flex, wrist_flex
        breath = self.breathing_amplitude * math.sin(2 * math.pi * self.breathing_freq_hz * t)
        for j in ("shoulder_lift", "elbow_flex", "wrist_flex"):
            joints[j] = joints[j] + breath

        # Micro-fidget: all joints, small sine with per-joint phase
        for j in JOINT_NAMES:
            phase = self._micro_phases[j]
            m = self.micro_fidget_amplitude * math.sin(
                2 * math.pi * self.micro_fidget_freq_hz * t + phase
            )
            if j == "gripper":
                joints[j] = max(0, min(100, joints[j] + m))
            else:
                joints[j] = joints[j] + m

        return joints
