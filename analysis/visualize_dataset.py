"""
Afterimage — Dataset Visualization
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation


dataset = LeRobotDataset("lerobot/pusht")

fig, ax = plt.subplots()
img = ax.imshow(dataset[0]['observation.image'].numpy().transpose(1, 2, 0))
height, width = dataset[0]['observation.image'].shape[1:]
ax.set_axis_off()

def update(i):
    data = dataset[i]['observation.image'].numpy().transpose(1, 2, 0)
    y, x = dataset[i]['observation.state']
    x, y = int(x * height / 512), int(y * width / 512)
    data[x, y] = [255, 0, 0]
    y, x = dataset[i]['action']
    x, y = int(x * height / 512), int(y * width / 512)
    data[x, y] = [0, 255, 0]
    img.set_data(data)
    ax.set_title(f"Episode {dataset[i]['episode_index']}, Frame {dataset[i]['frame_index']}, Reward {dataset[i]['next.reward']}, Success {dataset[i]['next.success']}")
    return [img]

anim = FuncAnimation(fig, update, frames=len(dataset), interval=100, blit=False)
plt.show()

# Keys:
# 'observation.state': coordinates of the robot in [512, 512]
# 'observation.image': image frame
# 'action': goal coordinates in range [512, 512]
# 'episode_index': index of the episode (max: 205)
# 'frame_index': index of the frame in the episode (typically max around 150)
# 'timestamp': timestamp of the frame in the episode (10 fps)
# 'next.reward': percentage of overlap
# 'next.done': whether the episode is done (episode max length reached)
# 'next.success': whether the episode was successful (95% overlap)
# 'index': index of the sample in the dataset (max: 25,649)
