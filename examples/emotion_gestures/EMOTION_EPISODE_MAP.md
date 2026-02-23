# Emotion ID → Episode Index Mapping

For programmatic recording with default keyframes, episodes are saved in this order:

| Episode | Emotion ID |
|---------|------------|
| 0       | neutral    |
| 1       | happy      |
| 2       | sad        |
| 3       | curious    |
| 4       | wave       |

When using `--emotions` to record a subset, the episode order follows the given list. Use `--dataset.emotion_id=<id>` with `play_emotion.py` to replay by name regardless of episode index.
