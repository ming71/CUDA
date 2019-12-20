import torch.nn as nn
import torch.nn.functional as F
from .ORConv import ORConv2d
from orn.functions import oraligned1d

class ORN(nn.Module):
    def __init__(self, nOrientation=8):
        super(ORN, self).__init__()
        self.nOrientation = nOrientation

        self.conv1 = ORConv2d(1, 10, arf_config=(1, nOrientation), kernel_size=3)
        self.conv2 = ORConv2d(10, 20, arf_config=nOrientation, kernel_size=3)
        self.conv3 = ORConv2d(20, 40, arf_config=nOrientation, kernel_size=3, stride=1, padding=1)
        self.conv4 = ORConv2d(40, 80, arf_config=nOrientation, kernel_size=3)

        self.fc1 = nn.Linear(640, 1024)
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))

        x = oraligned1d(x, self.nOrientation)

        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)