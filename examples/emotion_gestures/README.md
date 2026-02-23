# SO101 Emotion Gestures

Emotion gesture library for SO-Arm101 robot arm, inspired by the Pixar Luxo Jr. lamp. Stores gestures as LeRobotDataset v3.0 (no video) and plays them back via `lerobot-replay`-compatible logic.

## Quick Start

### Record (programmatic, no hardware)

```bash
cd lerobot
python examples/emotion_gestures/record_emotion.py --mode=programmatic \
    --dataset.repo_id=user/emotion_gestures \
    --dataset.root=./data/emotion_gestures \
    --emotions neutral happy sad wave
```

### Record (Leader arm teleoperation)

```bash
python examples/emotion_gestures/record_emotion.py --mode=leader \
    --robot.port=/dev/ttyACM0 \
    --teleop.port=/dev/ttyACM1 \
    --dataset.repo_id=user/emotion_gestures \
    --dataset.single_task=happy \
    --dataset.episode_time_s=15
```

### Play by episode index

```bash
python examples/emotion_gestures/play_emotion.py \
    --robot.port=/dev/ttyACM0 \
    --dataset.repo_id=user/emotion_gestures \
    --dataset.root=./data/emotion_gestures \
    --dataset.episode=0
```

### Play by emotion_id

```bash
python examples/emotion_gestures/play_emotion.py \
    --robot.port=/dev/ttyACM0 \
    --dataset.repo_id=user/emotion_gestures \
    --dataset.root=./data/emotion_gestures \
    --dataset.emotion_id=happy
```

## Using lerobot-replay

The dataset is compatible with `lerobot-replay`:

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{}' \
    --dataset.repo_id=user/emotion_gestures \
    --dataset.root=./data/emotion_gestures \
    --dataset.episode=0
```

## Emotion IDs

| ID        | Description (Luxo Jr.–style)        |
|-----------|-------------------------------------|
| neutral   | Rest pose                           |
| happy     | Lift, gentle sway, gripper open     |
| sad       | Head down, arm lowered              |
| curious   | Lean forward, slight tilt           |
| wave      | Shoulder swing, waving motion       |

## Dataset Format

LeRobotDataset v3.0, same as `lerobot-record` with `--dataset.video=false`:

- `meta/info.json` – features, fps, robot_type, no video_path
- `data/chunk-*/file-*.parquet` – state and action frames
- Each episode = one emotion; `task` field = emotion_id

## Future: Vision-Based Emotion Sync

Map visual emotion recognition outputs to emotion_id, then call `play_emotion.py --dataset.emotion_id=<detected>` or equivalent replay logic.
