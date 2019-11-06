import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, Fully
from data import get_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'fully':
        model = Fully()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    # List to hold loss and accuracy history
    Loss_hist = []
    Acc_hist = []
    Loss_val_hist, Acc_val_hist = [], []

    # Run any number of epochs you want
    ep = 14
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in tqdm(enumerate(train_loader,1)):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
        acc = correct_cnt / total_cnt
        ave_loss = total_loss / len(train_loader)
        Loss_hist.append((epoch, ave_loss))
        Acc_hist.append((epoch, acc))
        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        correct_val_cnt, total_val_loss, total_val_cnt = 0, 0, 0
        
        for batch, (x, label) in tqdm(enumerate(val_loader,1)):
            out = model(x)
            loss = criterion(out, label)
            total_val_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_val_cnt += x.size(0)
            correct_val_cnt += (pred_label == label).sum().item()
        acc_val = correct_val_cnt / total_val_cnt
        ave_loss_val = total_val_loss / len(val_loader)
        print ('Validating loss: {:.6f}, acc: {:.3f}'.format(
                     ave_loss_val, acc_val))
        Loss_val_hist.append((epoch,ave_loss_val))
        Acc_val_hist.append((epoch,acc_val))
        model.train()

    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Plot Learning Curve
    # TODO
    plt.figure(figsize=(10, 10))
    plt.plot(*zip(*Loss_hist))
    plt.savefig('results/loss_train_'+model_type+'.png')
    plt.clf()
    plt.plot(*zip(*Acc_hist))
    plt.savefig('results/acc_train_'+model_type+'.png')
    plt.clf()
    plt.plot(*zip(*Loss_val_hist))
    plt.savefig('results/loss_val_'+model_type+'.png')
    plt.clf()
    plt.plot(*zip(*Acc_val_hist))
    plt.savefig('results/acc_val_'+model_type+'.png')

  

