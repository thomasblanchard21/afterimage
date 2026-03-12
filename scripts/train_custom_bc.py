import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, Subset

class CustomBC(nn.Module):
    def __init__(self):
        super(CustomBC, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    run = wandb.init(project="afterimage")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomBC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = LeRobotDataset("lerobot/pusht")
    episode_indices = dataset.hf_dataset["episode_index"]
    train_indices = [i for i, ep in enumerate(episode_indices) if ep < 180]
    test_indices = [i for i, ep in enumerate(episode_indices) if ep >= 180]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    for epoch in range(5):
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            states = batch['observation.state'].to(device) / 512.0
            actions = batch['action'].to(device) / 512.0
            optimizer.zero_grad()
            pred = model(states)
            loss = nn.MSELoss()(pred, actions).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        run.log({"train_loss": epoch_loss / len(train_dataloader)})

    test_loss = 0.0
    for batch in tqdm(test_dataloader, desc="Testing"):
        states = batch['observation.state'].to(device) / 512.0
        actions = batch['action'].to(device) / 512.0
        with torch.no_grad():
            pred = model(states)
            loss = nn.MSELoss()(pred, actions).mean()
            test_loss += loss.item()
    run.log({"test_loss": test_loss / len(test_dataloader)})

    torch.save(model.state_dict(), "outputs/models/custom_bc.pth")

    run.finish()