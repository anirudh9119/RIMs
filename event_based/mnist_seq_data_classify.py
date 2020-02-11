
import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import scipy.misc


#for x,y in train_loader:
#    print(x.shape,y.shape)
#    print('x shape', x.shape)
#    xr = F.upsample(x, size=(14,14), mode='bilinear')
#    print('xr shape', xr.shape)
#    save_image(x, 'dsmnist.png')
#
#    print(x[0])
#    print(xr[0])

'''
Returns:
    x: (784,50000) int32.
    y: (784,50000) int32.
'''

def mnist_data():
    mnist_trainset = datasets.MNIST(root='/home/anirudh/blocks/sparse_relational/data', train=True, download=True, transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
    mnist_testset = datasets.MNIST(root='/home/anirudh/blocks/sparse_relational/data', train=False, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

    num_val = len(mnist_trainset) // 5
    np.random.seed(0)
    num_train = len(mnist_trainset)
    idxs = np.random.choice(num_train, num_train, replace=False)

    train_sampler = SubsetRandomSampler(idxs[num_val:])
    val_sampler = SubsetRandomSampler(idxs[:num_val])

    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size = 64, drop_last=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size = 64, drop_last=True, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size = 64, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader
