from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 1. 加载你本地存放在 .cache 里的数据
dataset = LeRobotDataset("seeedstudio123/task04")

# 2. 将数据重命名并推送到你的个人仓库
dataset.push_to_hub("kleinlau17/task04")
