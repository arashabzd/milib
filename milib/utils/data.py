import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def cifar10(path, batch_size, patch_size, stride, augment=None, download=True):
    num_patches = (32 - patch_size) // stride + 1    # number of patches in a row/column

    base_trans = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize([.5]*3, [.5]*3)])
    if augment is not None:
        train_trans = transforms.Compose([augment, base_trans])
    else:
        train_trans = base_trans

    trainset = datasets.CIFAR10(path, train=True, transform=train_trans, download=download)
    testset = datasets.CIFAR10(path, train=False, transform=base_trans, download=download)

    def patch_dataset(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        n = len(data)
        for k in range(n):
            data[k] = torch.stack(
                [
                    data[k][:, i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]
                    for i in range(num_patches) 
                    for j in range(num_patches)
                ]
            ) 
        data = torch.stack(data)
        data = data.reshape(n, num_patches, num_patches, 3, patch_size, patch_size)
        target = torch.LongTensor(target)
        return [data, target]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, collate_fn=patch_dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                             shuffle=False, collate_fn=patch_dataset)
    
    return trainloader, testloader