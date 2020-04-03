import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import nets.resnet as resnet
import mi.estim as estim
import mi.critics as critics
import utils.data as data
import models.cpc as cpc


def run(gpu_device, batch_size, patch_size, stride, p1, p2, epochs, log_interval):
#     print(gpu_device, batch_size, patch_size, stride, p1, p2, epochs, log_interval)

    if gpu_device >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_device))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
        torch.cuda.set_device(device)
    
    print('training on {}.'.format(device))
    
    trans = []
    trans.append(transforms.RandomGrayscale(.5))
    trans.append(transforms.RandomHorizontalFlip(.5))
    trans.append(transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)))
    augment = transforms.Compose(trans)
    
    num_patche = (32 - patch_size) // stride + 1
    trainloader, testloader = data.cifar10('../data/', batch_size, patch_size, stride, augment)
    
    encoder = cpc.Encoder(resnet.resnet20(patch_size))
    context = cpc.RNNContext(64, 64)
    critics1 = {str(p): critics.BiLinearCritic(64, 64) for p in range(p1, p2)}
    critics2 = {str(p): critics.BiLinearCritic(64, 64) for p in range(p1, p2)}
    model = cpc.CPC(encoder, context, critics1, critics2).to(device)
    infonce = estim.InfoNCE()
    adam = optim.Adam(model.parameters())
    
    model.train()
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0.
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            adam.zero_grad()
            scores = model(x)
            loss = infonce(scores[0]) 
            for score in scores[1:]:
                loss += infonce(score)
            loss.backward()
            adam.step()
            running_loss += loss.item()

            if i % log_interval == (log_interval - 1):
                print('\titeration {}: mi = {}'.format(i + 1, -running_loss/log_interval))
                running_loss = 0.

        torch.save(model.state_dict(), '../saved_models/cpc.chkpt')
    
    print('Encoder training finished.')
    print('\n Training classifier:\n')
    
    encoder.eval()
    classifier = nn.Linear(64, 10).to(device)
    adam = optim.Adam(classifier.parameters())
    ce = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        print('EPOCH {}:'.format(epoch + 1))
        classifier.train()
        running_loss = 0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            adam.zero_grad()
            with torch.no_grad():
                x = encoder(x)
                x = x.flatten(start_dim=1, end_dim=2)
                x = torch.mean(x, 1)
            y_h = classifier(x)
            loss = ce(y_h, y)
            loss.backward()
            adam.step()
            running_loss += loss.item()
            
            if i % log_interval == (log_interval - 1):
                print('\titeration {}: loss = {}'.format(i + 1, running_loss/log_interval))
                running_loss = 0
        
        if epoch % 5 == 4:
            correct = 0
            total = 0
            classifier.eval()
            with torch.no_grad():
                for (x, y) in testloader:
                    x, y = x.to(device), y.to(device)
                    x = encoder(x)
                    x = x.flatten(start_dim=1, end_dim=2)
                    x = torch.mean(x, 1)
                    y_h = classifier(x)
                    _, predicted = torch.max(y_h.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                
            print('Test Accuracy: {}'.format(100*correct/total))
            
                
        torch.save(classifier.state_dict(), '../saved_models/clf.chkpt')
                
    correct = 0
    total = 0
    classifier.eval()
    with torch.no_grad():
        for (x, y) in testloader:
            x, y = x.to(device), y.to(device)
            x = encoder(x)
            x = x.flatten(start_dim=1, end_dim=2)
            x = torch.mean(x, 1)
            y_h = classifier(x)
            _, predicted = torch.max(y_h.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print('Test Accuracy: {}'.format(100*correct/total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_device', type=int, default=-1,
                        help='gpu device (default: -1 (cpu))')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--patch_size', type=int, default=12,
                        help='patch size (default: 12)')
    parser.add_argument('--stride', type=int, default=4,
                        help='stride (default: 4)')
    parser.add_argument('--p1', type=int, default=3,
                        help='first prediction step (default: 3)')
    parser.add_argument('--p2', type=int, default=4,
                        help='last prediction step (default: 4)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    
    run(args.gpu_device, 
        args.batch_size, 
        args.patch_size, 
        args.stride, 
        args.p1, 
        args.p2, 
        args.epochs, 
        args.log_interval)