###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import alexnet
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from tqdm import tqdm
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_tsne(tsne_fig_path, features, targets):
    print("Plotting tsne graph...")
    X_embedded = TSNE(n_components=2).fit_transform(features)
    # print("X_embedded:", X_embedded)
    y = np.array([x.item() for x in targets])
    target = range(np.max(y)+1)
    plt.figure(figsize=(6, 5))
    colors = cm.rainbow(np.linspace(0, 1, len(target)))
    for i, c, label in zip(target, colors, y):
        plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], c=c.reshape((1,4)), label=label)
    plt.legend()
    plt.savefig(tsne_fig_path)

def main():
    data_dir, tsne_fig_path = sys.argv[1], sys.argv[2]

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    extractor = alexnet(pretrained=True).features
    extractor.eval()

    # Load datasets
    IF_train = ImageFolder(root = data_dir+'train', transform = image_transforms)
    IF_temp_val = ImageFolder(root = data_dir+'valid', transform = image_transforms)
    train_loader = DataLoader(IF_train, batch_size=1, num_workers=4, shuffle=True)
    # define lengths to split validation dataset into val and test
    val_length = int(len(IF_temp_val)*0.7)
    test_length = len(IF_temp_val) - val_length
    lengths = [val_length, test_length]
    # IF_val, IF_test = random_split(IF_temp_val, lengths)
    val_loader = DataLoader(IF_temp_val, batch_size=1, num_workers=4, shuffle=True)
    # test_loader = DataLoader(IF_test, batch_size=1, num_workers=4, shuffle=True)

    extracted_features = []
    targets = []
    print("Extracting features...")
    # Extract features using alexnet on train_loader
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        feat = extractor(data).view(data.size(0),256,-1)
        feat = torch.mean(feat,2)
        feat = feat.detach().numpy()
        extracted_features.append(feat.squeeze())
        targets.append(target)
    # print(extracted_features[0].shape)
    print("Features extracted")

    # Use n_components = 20
    pca = PCA(n_components=20, svd_solver='auto')
    pca.fit(extracted_features)
    extracted_features = pca.transform(extracted_features)
    # print(extracted_features[0].shape)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(extracted_features,targets)

    val_features = []
    val_targets = []
    first_ten_features = []
    first_ten_targets = []
    print("Iterating validation set...")
    for batch_idx, (data, target) in enumerate(val_loader):
        feat = extractor(data).view(data.size(0),256,-1)
        feat = torch.mean(feat,2)
        feat = feat.detach().numpy()
        val_features.append(feat.squeeze())
        val_targets.append(target)
        if target < 10:
            first_ten_features.append(feat.squeeze())
            first_ten_targets.append(target)
    print("Validation set iterated")
    val_features = pca.transform(val_features)
    # plot tsne
    plot_tsne(tsne_fig_path, first_ten_features, first_ten_targets)

    val_predicts = knn_model.predict(val_features)
    print("length of val_predicts:", len(val_predicts))
    total_predicts, corr_predicts = 0, 0
    for x in range(len(val_predicts)):
        if val_targets[x].item() == val_predicts[x]:
            corr_predicts += 1
        total_predicts += 1
    print("Total prediction:", total_predicts)
    print("Correct predictions:", corr_predicts)
    print("Test accuracy:", corr_predicts/total_predicts)

if __name__ == "__main__":
    main()


        



