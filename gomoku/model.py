from torch import nn

from config import BOARD_SIZE


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers =nn.Sequential(
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
            nn.Flatten(),
            nn.Linear(64*BOARD_SIZE*BOARD_SIZE, BOARD_SIZE*BOARD_SIZE)
        )


    def forward(self, x):
        logits = self.layers(x)
        return logits