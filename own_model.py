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

if __name__ == '__main__':
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    # Load datasets
    IF_train = ImageFolder(root = 'hw2-4_data/problem2/train', transform = image_transforms)
    IF_temp_val = ImageFolder(root = 'hw2-4_data/problem2/valid', transform = image_transforms)

    # define lengths to split validation dataset into val and test
    val_length = int(len(IF_temp_val)*0.7)
    test_length = len(IF_temp_val) - val_length
    lengths = [val_length, test_length]
    IF_val, IF_test = random_split(IF_temp_val, lengths)

    # Create DataLoaders
    train_loader = DataLoader(IF_train, batch_size=8, num_workers=4, shuffle=True)
    val_loader = DataLoader(IF_val, batch_size=8, num_workers=4, shuffle=True)
    test_loader = DataLoader(IF_test, batch_size=8, num_workers=4, shuffle=True)
    print("Length of training dataset:", len(train_loader))
    print("Length of validating dataset:", len(val_loader))
    print("Length of testing dataset:", len(test_loader))

    loss_fn = nn.CrossEntropyLoss() 

    # Load facenet model
    FN = FaceNet(phase='train')

    # Set the type of gradient optimizer and the model it update 
    #optimizer = optim.SGD(FN.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(FN.parameters(), lr=0.001, betas=[0.9,0.999])

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using Cuda")
        FN.cuda()

    # Record best validation accuracy to determine early stop
    best_val_acc = 0
    # Variable to record how many epochs not improving
    no_imp_ep = 0

    Loss_hist, Acc_hist, Loss_val_hist, Acc_val_hist = [], [], [], []
    # Number of epochs
    ep = 10
    for epoch in range(ep):
        total_prediction = np.zeros(100)
        # Determine whether to early stop
        if no_imp_ep >= 20:
            print("Early stop on epoch:", epoch)
            break

        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0

        for batch_idx, (data, target) in tqdm(enumerate(train_loader, 1)):

            optimizer.zero_grad()

            if use_cuda:
                data, target = data.cuda(), target.cuda()
            #print(target)
            # for testing model only
            #target = torch.LongTensor([10])
            out = FN(data)
            #print("out",out)
            loss = loss_fn(out, target)
            #print("loss:", loss)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            #print("predict label:", pred_label)
            #print("target:", target)
            total_cnt += data.size(0)
            total_prediction[pred_label.data.numpy()]+=1
            correct_cnt += (pred_label == target).sum().item()

            #print("total loss:", total_loss)
            # Show the training information
            
            if batch_idx % 500 == 0 or batch_idx == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch_idx           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch_idx, ave_loss, acc))
                print("total predictions:",total_prediction)
        acc = correct_cnt / total_cnt
        ave_loss = total_loss / len(train_loader)
        Loss_hist.append((epoch, ave_loss))
        Acc_hist.append((epoch, acc))

        if epoch % 5==0:
            # Save trained model
            torch.save(FN.state_dict(), './checkpoint/face/'+str(epoch)+'.pth')

        ################
        ## Validation ##
        ################
        FN.eval()
        # TODO
        correct_val_cnt, total_val_loss, total_val_cnt = 0, 0, 0
        
        for batch, (x, label) in tqdm(enumerate(val_loader,1)):
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            out = FN(x)
            loss = loss_fn(out, label)
            total_val_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_val_cnt += x.size(0)
            correct_val_cnt += (pred_label == label).sum().item()
        acc_val = correct_val_cnt / total_val_cnt

        # Update best val acc and reset no_imp_ep
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            no_imp_ep = 0
        else:
            no_imp_ep += 1

        ave_loss_val = total_val_loss / len(val_loader)
        print ('Validating loss: {:.6f}, acc: {:.3f}'.format(
                     ave_loss_val, acc_val))
        Loss_val_hist.append((epoch,ave_loss_val))
        Acc_val_hist.append((epoch,acc_val))
        FN.train()

        plt.figure(figsize=(10, 10))
        plt.plot(*zip(*Loss_hist))
        plt.savefig('results/face/loss_train.png')
        plt.clf()
        plt.plot(*zip(*Acc_hist))
        plt.savefig('results/face/acc_train.png')
        plt.clf()
        plt.plot(*zip(*Loss_val_hist))
        plt.savefig('results/face/loss_val.png')
        plt.clf()
        plt.plot(*zip(*Acc_val_hist))
        plt.savefig('results/face/acc_val.png')
    # Save trained model
    torch.save(FN.state_dict(), './checkpoint/face/final.pth')
    # Plot Learning Curve
    # TODO
    plt.figure(figsize=(10, 10))
    plt.plot(*zip(*Loss_hist))
    plt.savefig('results/face/loss_train.png')
    plt.clf()
    plt.plot(*zip(*Acc_hist))
    plt.savefig('results/face/acc_train.png')
    plt.clf()
    plt.plot(*zip(*Loss_val_hist))
    plt.savefig('results/face/loss_val.png')
    plt.clf()
    plt.plot(*zip(*Acc_val_hist))
    plt.savefig('results/face/acc_val.png')







