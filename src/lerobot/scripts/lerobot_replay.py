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

"""
Replays the actions of an episode from a dataset on a robot.

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

Replay the same episode multiple times:
```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --dataset.repo_id=aliberts/record-test \
    --dataset.episode=0 \
    --dataset.num_repeats=5
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30
    # Number of times to replay this episode. Default is 1.
    num_repeats: int = 1
    # Seconds to wait after the last frame before disconnecting, so the robot can finish the last action. Default is 2.0. Set to 0 to keep previous behavior.
    post_replay_delay: float = 2.0


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)

    robot.connect()

    try:
        num_repeats = cfg.dataset.num_repeats
        last_processed_action = None
        for repeat_idx in range(num_repeats):
            if repeat_idx == 0:
                log_say("Replaying episode", cfg.play_sounds, blocking=True)
            elif num_repeats > 1:
                logging.info("Repeat %d/%d", repeat_idx + 1, num_repeats)

            for idx in range(len(episode_frames)):
                start_episode_t = time.perf_counter()

                action_array = actions[idx][ACTION]
                action = {}
                for i, name in enumerate(dataset.features[ACTION]["names"]):
                    action[name] = action_array[i]

                robot_obs = robot.get_observation()

                processed_action = robot_action_processor((action, robot_obs))

                robot.send_action(processed_action)
                last_processed_action = processed_action

                dt_s = time.perf_counter() - start_episode_t
                precise_sleep(max(1 / dataset.fps - dt_s, 0.0))

        if cfg.dataset.post_replay_delay > 0:
            if last_processed_action is not None:
                logging.info(
                    "Holding last action for %.1fs before disconnect.",
                    cfg.dataset.post_replay_delay,
                )
                num_hold_steps = int(cfg.dataset.post_replay_delay * dataset.fps)
                for _ in range(num_hold_steps):
                    robot.send_action(last_processed_action)
                    precise_sleep(1.0 / dataset.fps)
            else:
                logging.info(
                    "Waiting %.1fs before disconnect.",
                    cfg.dataset.post_replay_delay,
                )
                time.sleep(cfg.dataset.post_replay_delay)
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
