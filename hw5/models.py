import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

class cifar10_resnet_50(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.resnet_50 = tmodels.resnet50(pretrained=pretrained)
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        output = self.resnet_50(x)
        output = self.classify(output)
        return output

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.process2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        output = self.process1(x)
        output = output.view(output.shape[0], -1)
        output = self.process2(output)
        return output