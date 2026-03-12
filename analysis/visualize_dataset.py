"""
Afterimage — Dataset Visualization

Explore the PushT dataset structure: actions, observations, episode lengths.
This is YOUR code to write. Start by loading the dataset and inspecting shapes.

Hints:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("lerobot/pusht")
    sample = dataset[0]
    print(sample.keys())
    print({k: v.shape for k, v in sample.items() if hasattr(v, 'shape')})
"""

# TODO: You write this — explore the dataset, plot action distributions,
# visualize a few episodes. This is where you build intuition for the data.
