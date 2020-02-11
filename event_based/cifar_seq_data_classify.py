import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image

#for x,y in train_loader:
#    print(x.shape,y.shape)
#    print('x shape', x.shape)
#    xr = F.upsample(x, size=(14,14), mode='bilinear')
#    print('xr shape', xr.shape)
#    save_image(x, 'dscifar.png')
#
#    print(x[0])
#    print(xr[0])

'''
Returns:
    x: (784,50000) int32.
    y: (784,50000) int32.
'''

def cifar_data():
    transform = transforms.Compose(
        [transforms.ToTensor()])

    cifar_trainset = datasets.CIFAR10(root='/home/anirudh/blocks/sparse_relational/data/', train=True, download=True, transform = transform)
    cifar_testset = datasets.CIFAR10(root='/home/anirudh/blocks/sparse_relational/data/', train=False, download=True, transform = transform)
    noise_set = torch.tensor(np.random.randn(10000, 3, 32, 32), dtype=torch.float)

    num_val = len(cifar_trainset) // 5
    num_train = len(cifar_trainset)
    idxs = np.random.choice(num_train, num_train, replace=False)

    train_sampler = SubsetRandomSampler(idxs[num_val:])
    val_sampler = SubsetRandomSampler(idxs[:num_val])

    train_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size = 64, drop_last=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size = 64, drop_last=True, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(cifar_testset, batch_size = 64, shuffle=False, drop_last=True)
    noise_loader = torch.utils.data.DataLoader(noise_set, batch_size = 64, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, noise_loader
