import torch.utils.data as data
import torchvision.transforms as transforms
import os

class faceDataset(data.Dataset):

    def __init__(self, rootdir, phase):
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        self.phase = phase
        self.images = []
        self.labels = []
        self.full_paths = []
        if self.phase == 'train':
            self.datadir = rootdir+'/train'
        else:
            self.datadir = rootdir+'/valid'
        for root, subdir, fnames in sorted(os.walk(self.datadir)):
            for fname in fnames:
                full_name = os.path.join(root,fname)
                print(full_name)

    def __len__(self):
        return

    def __getitem__(self, idx):
        return
if __name__ == '__main__':
    fd = faceDataset('hw2-4_data/problem2','train')
