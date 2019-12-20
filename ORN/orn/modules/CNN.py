import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=3)
        self.conv2 = nn.Conv2d(80, 160, kernel_size=3)
        self.conv3 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(320, 640, kernel_size=3)

        self.fc1 = nn.Linear(640, 1024)
        self.fc2 = nn.Linear(1024, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

