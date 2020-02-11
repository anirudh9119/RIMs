import numpy as np
import torch
from torch.utils import data

def add_data(size, seq_len, batch_size, shuffle=False, drop_last=False):
    nums = np.random.uniform(low=0, high=1, size=[size, seq_len])
    i1 = np.random.choice(seq_len//2, size)
    i2 = seq_len//2 + np.random.choice(seq_len//2, size)
    outputs = torch.tensor(nums[np.arange(size),i1] + nums[np.arange(size),i2]).float()

    one_hot_i1 = np.zeros([size, seq_len])
    one_hot_i1[np.arange(size), i1] = 1

    one_hot_i2 = np.zeros([size, seq_len])
    one_hot_i2[np.arange(size), i2] = 1

    nums = np.reshape(nums, [size, seq_len, 1])
    one_hot_i1 = np.reshape(one_hot_i1, [size, seq_len, 1])
    one_hot_i2 = np.reshape(one_hot_i2, [size, seq_len, 1])

    inputs = torch.tensor(np.concatenate([nums, one_hot_i1, one_hot_i2], axis=2)).float()

    dataset = data.TensorDataset(inputs, outputs)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def adding_dataset(sizes, seq_lens, batch_size):
    assert(seq_lens[1] == seq_lens[2])

    train_loader = add_data(sizes[0], seq_lens[0], batch_size, True, True)
    val_loader = add_data(sizes[1], seq_lens[1], batch_size)
    test_loader = add_data(sizes[2], seq_lens[2], batch_size)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    x,y = adding_data(T=10)

    print(x[0,:,0])
    print(y[0,:,0])

    print(x.shape, y.shape)

