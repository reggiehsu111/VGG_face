from sklearn.manifold import TSNE
from torchvision.models import alexnet
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
import matplotlib.cm as cm
import os

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])

def load_model(state_dict_path, model):
    print("Loading state dict")
    model.load_state_dict(torch.load(state_dict_path))
    print("State dict loaded")

def read_images(directory):
    sub_dir_list = os.listdir(directory)
    first_ten_x = []
    first_ten_y = []
    for class_dir in sub_dir_list:
        class_num = class_dir.split('_')[1]
        if int(class_dir.split('_')[1]) <= 9:
            for image in os.listdir(directory+'/'+class_dir):
                full_path = os.path.join(directory,class_dir,image)
                first_ten_x.append(image_transforms(plt.imread(full_path)))
                first_ten_y.append(int(class_num))
    return first_ten_x, first_ten_y

def plot_tsne(tsne_fig_path, features, targets):
    print("Plotting tsne graph...")
    X_embedded = TSNE(n_components=2).fit_transform(features)
    # print("X_embedded:", X_embedded)
    y = np.array(targets)
    target = range(np.max(y)+1)
    plt.figure(figsize=(6, 5))
    colors = cm.rainbow(np.linspace(0, 1, len(target)))
    for i, c in zip(target, colors):
        plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], c=c.reshape((1,4)), label=i)
    plt.legend()
    plt.savefig(tsne_fig_path)

if __name__ == '__main__':
    first_ten_x, first_ten_y = read_images('hw2-4_data/problem2/valid')
    
    # Load facenet model
    FN = FaceNet(phase='test')
    load_model('checkpoint/face/final.pth',FN)
    extractor = FN.features
    extractor.eval()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using Cuda")
        extractor.cuda()

    extracted_features = []
    for x in first_ten_x:
        x = x.unsqueeze(0)
        feat = extractor(x).view(x.size(0),128,-1)
        feat = feat.mean(2)
        feat = feat.detach().numpy()
        extracted_features.append(feat.squeeze())

    plot_tsne("own_model_tsne.png", extracted_features, first_ten_y)
