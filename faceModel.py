import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

class FaceNet(nn.Module):
    def __init__(self, phase):
        super(FaceNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.25),
        )
        # self.conv1 = nn.Conv2d()
        self.classifier = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Linear(128*576 , 320),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.25),
            nn.Linear(320, 100),
            #nn.Softmax(dim=1)
        )
        # init weights 
        #self.features.apply(self.init_weights)
        #self.classifier.apply(self.init_weights)

    def init_weights(self,m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.001)


    def forward(self, x):
        x = self.features(x).view(x.size(0),-1)
        #print(x.shape)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    fn = FaceNet('train')
