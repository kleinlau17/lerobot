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

## Edit / Teach Emotion Keyframes

You can edit or create emotion keyframes in a small teaching CLI, then use them
both in `robot_viewer` and for programmatic recording.

### 1. Interactive teaching CLI (no hardware required)

```bash
cd lerobot
python examples/emotion_gestures/cli_teach_emotions.py \
    --output-json examples/emotion_gestures/emotion_actions_taught.json
```

This will open an interactive menu where you can:

- list / create / rename / delete emotions and actions
- record keyframes by manually entering joint values (degrees, gripper 0–100)
- adjust keyframe times `t ∈ [0, 1]` or delete them
- export to a JSON file compatible with `KeyframePlugin` in `robot_viewer`

### 2. Interactive teaching CLI with real SO101 follower (drag teaching)

If you want to generate or edit emotion keyframes by **physically dragging the
SO101 follower arm** and sampling its current pose, you can use:

```bash
cd lerobot
python examples/emotion_gestures/cli_teach_emotions.py \
    --mode=robot \
    --robot.port=/dev/ttyACM0 \
    --output-json examples/emotion_gestures/emotion_actions_robot.json
```

Workflow:

- put the follower arm into a safe, low-torque \"teaching\" configuration
  (so that you can move it by hand);
- in the CLI, create/select an emotion and action;
- physically move the arm to the desired pose, then choose
  \"录制一个 keyframe（从当前姿态）\" in the menu;
- repeat to build a sequence of keyframes; with `t=None`, the tool will
  automatically spread keyframe times uniformly in \[0, 1\];
- export to JSON and optionally import it back into `emotion_actions.py`
  using `import_emotion_json.py`, or load it in `robot_viewer`.

### 3. Load taught keyframes into record_emotion.py

You can use the exported JSON when generating programmatic episodes:

```bash
cd lerobot
python examples/emotion_gestures/record_emotion.py --mode=programmatic \
    --dataset.repo_id=user/emotion_gestures \
    --dataset.root=./data/emotion_gestures \
    --emotions neutral happy sad wave \
    --keyframes_json examples/emotion_gestures/emotion_actions_taught.json
```

If `--keyframes_json` is set, `record_programmatic` will flatten the JSON
structure (emotion → action → keyframes) into a per-emotion keyframe sequence
for dataset writing. If loading fails, it falls back to the built-in
`DEFAULT_KEYFRAMES`.

## Living Behavior (Pixar-style Idle)

Run continuous living behavior with a state machine that switches between idle (breathing + micro-motion) and emotions. Does not depend on record_emotion or play_emotion.

```bash
cd lerobot
python examples/emotion_gestures/run_living.py --robot.port=/dev/ttyACM0
```

Options:

- `--breathing.amplitude`, `--breathing.freq_hz` – idle breathing
- `--expression.interval_min_s`, `--expression.interval_max_s` – idle duration before next emotion (default 30–90 s)
- `--emotion.duration_s` – duration of each emotion animation (default 3 s)
- `--micro_fidget.amplitude` – micro-motion amplitude

State machine: `idle` (breathing + micro-fidget) ↔ `happy` / `curious` / `wave` / `sad`. Transitions to an emotion after the idle interval; returns to idle when the emotion animation ends. Supports `trigger_emotion(emotion)` for external triggers (e.g. vision, voice).

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
