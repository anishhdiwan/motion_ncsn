import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial


class DummyNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        # self.norm = ConditionalInstanceNorm2d
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        # self.act = act = nn.ReLU(True)

        self.linear1 = nn.Linear(5, 5)


    def forward(self, x, y):

        output = self.linear1(x)

        return output