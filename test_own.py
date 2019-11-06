import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.dataset import random_split
from faceModel import FaceNet
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

TOTAL_CLASSES = 100

def load_model(state_dict_path, model):
	print("Loading state dict")
	model.load_state_dict(torch.load(state_dict_path))
	print("State dict loaded")

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])
# Load datasets
IF_temp_val = ImageFolder(root = 'hw2-4_data/problem2/valid', transform = image_transforms)

# define lengths to split validation dataset into val and test
val_length = int(len(IF_temp_val)*0.7)
test_length = len(IF_temp_val) - val_length
lengths = [val_length, test_length]
IF_val, IF_test = random_split(IF_temp_val, lengths)

# Create DataLoaders
val_loader = DataLoader(IF_val, batch_size=8, num_workers=4, shuffle=True)
test_loader = DataLoader(IF_test, batch_size=8, num_workers=4, shuffle=True)
print("Length of validating dataset:", len(val_loader))
print("Length of testing dataset:", len(test_loader))

loss_fn = nn.CrossEntropyLoss() 

# Load facenet model
FN = FaceNet(phase='test')
load_model('checkpoint/face/final.pth',FN)

# Check if GPU is available, otherwise CPU is used
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using Cuda")
    FN.cuda()

FN.eval()

correct_test_cnt, total_test_loss, total_test_cnt = 0, 0, 0

for batch, (x, label) in tqdm(enumerate(test_loader,1)):
    if use_cuda:
        x, label = x.cuda(), label.cuda()
    out = FN(x)
    _, pred_label = torch.max(out, 1)
    total_test_cnt += x.size(0)
    correct_test_cnt += (pred_label == label).sum().item()
acc_test = correct_test_cnt / total_test_cnt
print("Testing accuracy:", acc_test)