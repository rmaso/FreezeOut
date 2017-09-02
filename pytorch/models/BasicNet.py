import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import utils
import math
from torch.autograd import Variable


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # input is 28x28
    # padding=2 for same padding
    self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
    # feature map size is 14*14 by pooling
    # padding=2 for same padding
    self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
    # feature map size is 7*7 by pooling
    self.fc = nn.Linear(64*7*7, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 64*7*7)   # reshape Variable
    x = self.fc(x)
    return F.log_softmax(x)
