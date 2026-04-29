from torch import nn

from config import BOARD_SIZE


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE)
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*BOARD_SIZE*BOARD_SIZE, 1),
            nn.Tanh() # keeping the output range [-1, 1]
        )


    def forward(self, x):
        features = self.backbone(x)
        return self.policy_head(features), self.value_head(features)
