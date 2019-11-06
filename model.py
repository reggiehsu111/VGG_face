import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(1,6,5,1)
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.drop1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.drop2 = nn.Dropout(0.25)
        self.linear1 = nn.Linear(16*4*4,120)
        self.linear2 = nn.Linear(120,84)
        self.drop3 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(84,10)
        self.module1 = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.mp1,
            self.drop1,
            self.conv2,
            nn.ReLU(),
            self.mp2,
            self.drop2
            )
        self.module2 = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.drop3,
            self.linear3,
            nn.Softmax()
            )


    def forward(self, x):
        out = self.module1(x)
        out = out.view(-1,16*4*4)
        out = self.module2(out)
        
        # TODO
        return out

    def name(self):
        return "ConvNet"

class Fully(nn.Module):
    def __init__(self):
        super(Fully, self).__init__()
        # TODO
        self.module = nn.Sequential(
            nn.Linear(784,500),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(500,250),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(250,32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,10),
            nn.Softmax()
            )

    def forward(self, x):
        x = x.view(x.size(0),-1) # flatten input tensor
        out = self.module(x)
        # TODO
        return out

    def name(self):
        return "Fully"

