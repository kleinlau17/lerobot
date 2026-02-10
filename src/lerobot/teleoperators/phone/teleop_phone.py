#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Docs:
# hebi: https://docs.hebi.us/tools.html#mobile-io
# teleop: https://github.com/SpesRobotics/teleop

import logging
import threading
import time

import hebi
import numpy as np
from teleop import Teleop

from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation

logger = logging.getLogger(__name__)


class BasePhone:
    _enabled: bool = False
    _calib_pos: np.ndarray | None = None
    _calib_rot_inv: Rotation | None = None

    def _reapply_position_calibration(self, pos: np.ndarray) -> None:
        self._calib_pos = pos.copy()

    @property
    def is_calibrated(self) -> bool:
        return (self._calib_pos is not None) and (self._calib_rot_inv is not None)

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "phone.pos": np.ndarray,  # shape (3,)
            "phone.rot": Rotation,  # scipy.spatial.transform.Rotation
            "phone.raw_inputs": dict,  # analogs/buttons or webXR meta
            "phone.enabled": bool,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        # No haptic or other feedback implemented yet
        pass

    def configure(self) -> None:
        # No additional configuration required for phone teleop
        pass

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # We could add haptic feedback (vibrations) here, but it's not implemented yet
        raise NotImplementedError


class IOSPhone(BasePhone, Teleoperator):
    name = "ios_phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self._group = None

    @property
    def is_connected(self) -> bool:
        return self._group is not None

    @check_if_already_connected
    def connect(self) -> None:
        logger.info("Connecting to IPhone, make sure to open the HEBI Mobile I/O app.")
        lookup = hebi.Lookup()
        time.sleep(2.0)
        group = lookup.get_group_from_names(["HEBI"], ["mobileIO"])
        if group is None:
            raise RuntimeError("Mobile I/O not found — check name/family settings in the app.")
        self._group = group
        logger.info(f"{self} connected to HEBI group with {group.size} module(s).")

        self.calibrate()

    def calibrate(self) -> None:
        # Check connection status
        if self._group is None:
            print("ERROR: Not connected to HEBI group!")
            return
        print(f"DEBUG: Connected to HEBI group with {self._group.size} module(s)")
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        button_num = self.config.calibration_button
        print(f"Press and hold B{button_num} in the HEBI Mobile I/O app to capture this pose...")
        print("\n[INFO] Troubleshooting tips:")
        print(f"  - Make sure B{button_num} button is configured in HEBI Mobile I/O app")
        print(f"  - Try pressing and HOLDING B{button_num} (not just tapping)")
        print("  - Check if any other buttons (B2-B8) respond when pressed")
        print("  - The debug output will show all button value changes")
        print(f"  - If B{button_num} doesn't work, you can change calibration_button in PhoneConfig\n")
        position, rotation = self._wait_for_capture_trigger()
        self._calib_pos = position.copy()
        self._calib_rot_inv = rotation.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        Blocks execution until the calibration trigger is detected from the iOS device.

        This method enters a loop, continuously reading the phone's state. It waits for the user to press
        and hold the configured button (default B1, configurable via calibration_button) in the HEBI Mobile I/O app.
        Once the button is pressed, the loop breaks and returns the phone's pose at that exact moment.

        Returns:
            A tuple containing the position (np.ndarray) and rotation (Rotation) of the phone at the
            moment the trigger was activated.
        """
        iteration = 0
        last_status_time = time.time()
        STATUS_INTERVAL = 1.0  # Print status every 1 second
        button_num = self.config.calibration_button
        last_button_value = None  # Track button value changes
        last_all_button_values = {}  # Track all button values for change detection
        
        while True:
            iteration += 1
            has_pose, position, rotation, fb_pose = self._read_current_pose()
            
            # Print detailed status periodically (every ~1 second)
            current_time = time.time()
            if current_time - last_status_time >= STATUS_INTERVAL:
                print(f"\n[DEBUG] Iteration {iteration} (waiting for B{button_num} press):")
                print(f"  - has_pose: {has_pose}")
                
                if fb_pose is not None:
                    # Check ARKit data
                    ar_pos = getattr(fb_pose, "ar_position", None)
                    ar_quat = getattr(fb_pose, "ar_orientation", None)
                    print(f"  - ar_position available: {ar_pos is not None}")
                    print(f"  - ar_orientation available: {ar_quat is not None}")
                    if ar_pos is not None:
                        print(f"  - ar_position: {ar_pos}")
                    
                    # Check IO object
                    io = getattr(fb_pose, "io", None)
                    print(f"  - io object available: {io is not None}")
                    
                    if io is not None:
                        button_b = getattr(io, "b", None)
                        button_a = getattr(io, "a", None)
                        print(f"  - io.b (button bank) available: {button_b is not None}")
                        print(f"  - io.a (analog bank) available: {button_a is not None}")
                        
                        # Try to read all button channels
                        if button_b is not None:
                            button_values = {}
                            for ch in range(1, 9):
                                try:
                                    if hasattr(button_b, "has_int") and button_b.has_int(ch):
                                        val = button_b.get_int(ch)
                                        button_values[f"b{ch}"] = val
                                    elif hasattr(button_b, "has_bool") and button_b.has_bool(ch):
                                        val = button_b.get_bool(ch)
                                        button_values[f"b{ch}"] = int(val)
                                    else:
                                        # Try direct access even if has_int/has_bool returns False
                                        try:
                                            val = button_b.get_int(ch)
                                            button_values[f"b{ch}"] = val
                                        except Exception:
                                            try:
                                                val = button_b.get_bool(ch)
                                                button_values[f"b{ch}"] = int(val)
                                            except Exception:
                                                pass  # Channel not available
                                except Exception as e:
                                    button_values[f"b{ch}"] = f"ERROR: {e}"
                            print(f"  - Button values: {button_values}")
                            # Always try to read the calibration button specifically for detailed info
                            try:
                                btn_val = None
                                if hasattr(button_b, "has_int") and button_b.has_int(button_num):
                                    btn_val = button_b.get_int(button_num)
                                elif hasattr(button_b, "has_bool") and button_b.has_bool(button_num):
                                    btn_val = button_b.get_bool(button_num)
                                else:
                                    try:
                                        btn_val = button_b.get_int(button_num)
                                    except Exception:
                                        try:
                                            btn_val = button_b.get_bool(button_num)
                                        except Exception:
                                            btn_val = "NOT_AVAILABLE"
                                print(f"  - B{button_num} button value: {btn_val} (type: {type(btn_val)})")
                            except Exception as e:
                                print(f"  - B{button_num} button read error: {e}")
                        
                        # Try to read analog channels
                        if button_a is not None:
                            analog_values = {}
                            for ch in range(1, 9):
                                try:
                                    if hasattr(button_a, "has_float") and button_a.has_float(ch):
                                        val = button_a.get_float(ch)
                                        analog_values[f"a{ch}"] = val
                                except Exception as e:
                                    analog_values[f"a{ch}"] = f"ERROR: {e}"
                            if analog_values:
                                print(f"  - Analog values: {analog_values}")
                    else:
                        print(f"  - fb_pose attributes: {dir(fb_pose)}")
                else:
                    print(f"  - fb_pose is None")
                
                last_status_time = current_time
            
            if not has_pose:
                time.sleep(0.01)
                continue

            io = getattr(fb_pose, "io", None)
            button_b = getattr(io, "b", None) if io is not None else None
            button_pressed = False
            current_button_value = None
            current_all_button_values = {}
            
            if button_b is not None:
                # Read all button values to detect any changes
                for ch in range(1, 9):
                    try:
                        if hasattr(button_b, "has_int") and button_b.has_int(ch):
                            current_all_button_values[f"b{ch}"] = button_b.get_int(ch)
                        elif hasattr(button_b, "has_bool") and button_b.has_bool(ch):
                            current_all_button_values[f"b{ch}"] = int(button_b.get_bool(ch))
                        else:
                            try:
                                current_all_button_values[f"b{ch}"] = button_b.get_int(ch)
                            except Exception:
                                try:
                                    current_all_button_values[f"b{ch}"] = int(button_b.get_bool(ch))
                                except Exception:
                                    pass
                    except Exception:
                        pass
                
                # Check for any button value changes
                button_changes = []
                for btn_name, btn_val in current_all_button_values.items():
                    last_val = last_all_button_values.get(btn_name)
                    if last_val is not None and btn_val != last_val:
                        button_changes.append(f"{btn_name}: {last_val} -> {btn_val}")
                
                if button_changes:
                    print(f"\n[DEBUG] Button value changes detected at iteration {iteration}:")
                    for change in button_changes:
                        print(f"  - {change}")
                
                # Update last values
                last_all_button_values = current_all_button_values.copy()
                
                # Try multiple methods to read the calibration button
                try:
                    # Method 1: Check if has_int exists and channel is available
                    if hasattr(button_b, "has_int") and button_b.has_int(button_num):
                        current_button_value = button_b.get_int(button_num)
                        button_pressed = bool(current_button_value)
                    # Method 2: Try has_bool
                    elif hasattr(button_b, "has_bool") and button_b.has_bool(button_num):
                        current_button_value = button_b.get_bool(button_num)
                        button_pressed = bool(current_button_value)
                    # Method 3: Try direct access (might work even if has_int returns False)
                    else:
                        try:
                            current_button_value = button_b.get_int(button_num)
                            button_pressed = bool(current_button_value)
                        except Exception:
                            try:
                                current_button_value = button_b.get_bool(button_num)
                                button_pressed = bool(current_button_value)
                            except Exception:
                                pass
                except Exception as e:
                    # Last resort: try direct access without checking
                    try:
                        current_button_value = button_b.get_int(button_num)
                        button_pressed = bool(current_button_value)
                    except Exception:
                        try:
                            current_button_value = button_b.get_bool(button_num)
                            button_pressed = bool(current_button_value)
                        except Exception:
                            pass
                
                # Debug: print button status when value changes
                if current_button_value != last_button_value:
                    print(f"\n[DEBUG] B{button_num} button value changed: {last_button_value} -> {current_button_value} (pressed={button_pressed}) at iteration {iteration}")
                    last_button_value = current_button_value
            
            if button_pressed:
                print(f"\n[SUCCESS] B{button_num} button pressed detected at iteration {iteration}!")
                return position, rotation

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """
        Reads the instantaneous 6-DoF pose from the connected iOS device via the HEBI SDK.

        This method fetches the latest feedback packet from the HEBI group, extracts the ARKit
        position and orientation, and converts them into a standard format. It also applies a
        configured camera offset to adjust the pose from the camera's frame to the phone's
        physical frame.

        Returns:
            A tuple containing:
            - A boolean indicating if a valid pose was successfully read.
            - The 3D position as a NumPy array, or None if not available.
            - The orientation as a `Rotation` object, or None if not available.
            - The raw HEBI feedback object for accessing other data like button presses.
        """
        fbk = self._group.get_next_feedback()
        if fbk is None:
            return False, None, None, None
        # Check if GroupFeedback has any modules
        try:
            if hasattr(fbk, "size") and fbk.size == 0:
                return False, None, None, None
            pose = fbk[0]
        except (IndexError, TypeError) as e:
            return False, None, None, None
        ar_pos = getattr(pose, "ar_position", None)
        ar_quat = getattr(pose, "ar_orientation", None)
        if ar_pos is None or ar_quat is None:
            return False, None, None, None
        # HEBI provides orientation in w, x, y, z format.
        # Scipy's Rotation expects x, y, z, w.
        quat_xyzw = np.concatenate((ar_quat[1:], [ar_quat[0]]))  # wxyz to xyzw
        rot = Rotation.from_quat(quat_xyzw)
        pos = ar_pos - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    @check_if_not_connected
    def get_action(self) -> dict:
        has_pose, raw_position, raw_rotation, fb_pose = self._read_current_pose()
        if not has_pose or not self.is_calibrated:
            return {}

        # Collect raw inputs (digital buttons / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
        io = getattr(fb_pose, "io", None)
        if io is not None:
            bank_a, bank_b = io.a, io.b
            if bank_a:
                for ch in range(1, 9):
                    if bank_a.has_float(ch):
                        raw_inputs[f"a{ch}"] = float(bank_a.get_float(ch))
            if bank_b:
                for ch in range(1, 9):
                    if bank_b.has_int(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_int(ch))
                    elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_bool(ch))

        # Use the configured calibration/enable button (default: B1) as the teleoperation enable gate
        button_num = getattr(self.config, "calibration_button", 1)
        enable = bool(raw_inputs.get(f"b{button_num}", 0))

        # Rising edge then re-capture calibration immediately from current raw pose
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_position)

        # Apply calibration
        pos_cal = self._calib_rot_inv.apply(raw_position - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rotation

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._group = None


class AndroidPhone(BasePhone, Teleoperator):
    name = "android_phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._android_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._teleop is not None

    @check_if_already_connected
    def connect(self) -> None:
        logger.info("Starting teleop stream for Android...")
        self._teleop = Teleop()
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        logger.info(f"{self} connected, teleop stream started.")

        self.calibrate()

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Touch and move on the WebXR page to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        Blocks execution until the calibration trigger is detected from the Android device.

        This method enters a loop, continuously checking the latest message received from the WebXR
        session. It waits for the user to touch and move their finger on the screen, which generates
        a `move` event. Once this event is detected, the loop breaks and returns the phone's current
        pose.

        Returns:
            A tuple containing the position (np.ndarray) and rotation (Rotation) of the phone at the
            moment the trigger was activated.
        """
        while True:
            with self._android_lock:
                msg = self._latest_message or {}

            if bool(msg.get("move", False)):
                ok, pos, rot, _pose = self._read_current_pose()
                if ok:
                    return pos, rot

            time.sleep(0.01)

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """
        Reads the latest 6-DoF pose received from the Android device's WebXR session.

        This method accesses the most recent pose data stored by the `_android_callback`. It uses a
        thread lock to safely read the shared `_latest_pose` variable. The pose, a 4x4 matrix, is
        then decomposed into position and rotation, and the configured camera offset is applied.

        Returns:
            A tuple containing:
            - A boolean indicating if a valid pose was available.
            - The 3D position as a NumPy array, or None if no pose has been received yet.
            - The orientation as a `Rotation` object, or None if no pose has been received.
            - The raw 4x4 pose matrix as received from the teleop stream.
        """
        with self._android_lock:
            if self._latest_pose is None:
                return False, None, None, None
            p = self._latest_pose.copy()
            pose = self._latest_pose
        rot = Rotation.from_matrix(p[:3, :3])
        pos = p[:3, 3] - rot.apply(self.config.camera_offset)
        return True, pos, rot, pose

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        """
        Callback function to handle incoming data from the Android teleop stream.

        This method is executed by the `teleop` package's subscriber thread whenever a new
        pose and message are received from the WebXR session on the Android phone. It updates
        the internal state (`_latest_pose` and `_latest_message`) with the new data.
        A thread lock is used to ensure that these shared variables are updated atomically,
        preventing race conditions with the main thread that reads them.

        Args:
            pose: A 4x4 NumPy array representing the phone's transformation matrix.
            message: A dictionary containing additional data, such as button presses or touch events.
        """
        with self._android_lock:
            self._latest_pose = pose
            self._latest_message = message

    @check_if_not_connected
    def get_action(self) -> dict:
        ok, raw_pos, raw_rot, pose = self._read_current_pose()
        if not ok or not self.is_calibrated:
            return {}

        # Collect raw inputs (B1 / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
        msg = self._latest_message or {}
        raw_inputs["move"] = bool(msg.get("move", False))
        raw_inputs["scale"] = float(msg.get("scale", 1.0))
        raw_inputs["reservedButtonA"] = bool(msg.get("reservedButtonA", False))
        raw_inputs["reservedButtonB"] = bool(msg.get("reservedButtonB", False))

        enable = bool(raw_inputs.get("move", False))

        # Rising edge then re-capture calibration immediately from current raw pose
        if enable and not self._enabled:
            self._reapply_position_calibration(raw_pos)

        # Apply calibration
        pos_cal = self._calib_rot_inv.apply(raw_pos - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rot

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
            self._latest_pose = None


class Phone(Teleoperator):
    """
    Phone-based teleoperator using ARKit (iOS via HEBI Mobile I/O App) or the teleop Python package (Android via WebXR API).
    For HEBI Mobile I/O we also expose 8 analog (a1-a8) and 8 digital (b1-b8) inputs.

    Press and hold **B1** to enable teleoperation. While enabled, the first B1 press
    captures a reference pose and rotation, when disabled and pressed again the position is reapplied.
    """

    config_class = PhoneConfig
    name = "phone"

    def __init__(self, config: PhoneConfig):
        super().__init__(config)
        self.config = config

        self._phone_impl: Teleoperator

        if self.config.phone_os == PhoneOS.IOS:
            self._phone_impl = IOSPhone(config)
        elif self.config.phone_os == PhoneOS.ANDROID:
            self._phone_impl = AndroidPhone(config)
        else:
            raise ValueError(f"Invalid config phone_os: {self.config.phone_os}")

    @property
    def is_connected(self) -> bool:
        return self._phone_impl.is_connected

    def connect(self) -> None:
        return self._phone_impl.connect()

    def calibrate(self) -> None:
        return self._phone_impl.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._phone_impl.is_calibrated

    @property
    def action_features(self) -> dict[str, type]:
        return self._phone_impl.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return self._phone_impl.feedback_features

    def configure(self) -> None:
        return self._phone_impl.configure()

    def get_action(self) -> dict:
        return self._phone_impl.get_action()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return self._phone_impl.send_feedback(feedback)

    def disconnect(self) -> None:
        return self._phone_impl.disconnect()
