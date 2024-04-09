import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

class TSNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(3748)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(3744)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.lin1 = nn.Linear(in_features=256, out_features=24)
        self.dropout = nn.Dropout(p=0.1)
        self.lin2 = nn.Linear(in_features=24, out_features=4)
        self.lin3 = nn.Linear(in_features=4, out_features=1)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x1 = torch.mean(x, dim=1)
        x2 = torch.max(x, dim=1).values
        x = torch.concat([x1, x2], axis=-1)
        x = F.leaky_relu(self.lin1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.lin2(x))
        x = F.sigmoid(self.lin3(x))
        return x