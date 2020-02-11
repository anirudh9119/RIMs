import numpy
import torch
from torch.utils import data

def copying_data(T=30, n_data=300*64, n_sequence=10, batch_size=64, make_rand=False, shuffle=False, drop_last=False):
    seq = numpy.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = numpy.zeros((n_data, T-1))
    zeros2 = numpy.zeros((n_data, T))
    marker = 9 * numpy.ones((n_data, 1))
    zeros3 = numpy.zeros((n_data, n_sequence))

    x = numpy.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = numpy.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')

    x = x.reshape(x.shape[0] // batch_size, batch_size, x.shape[1])
    y = y.reshape(y.shape[0] // batch_size, batch_size, y.shape[1])
    x = numpy.swapaxes(x, 1,2).astype('int64')
    y = numpy.swapaxes(y, 1,2).astype('int64')

    x = torch.from_numpy(x)#.float()
    y = torch.from_numpy(y)#.float()

    #if make_rand:
    #    x = torch.randint_like(x, 0,10)
    #    y = torch.randint_like(y, 0,10)
    #dataset = data.TensorDataset(x, y)
    #dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return x, y #dataloader

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

def copying_dataset(sizes, seq_lens, batch_size):
    assert(seq_lens[1] == seq_lens[2])

    train_loader = copying_data(sizes[0], seq_lens[0], batch_size, True, True)
    val_loader = copying_data(sizes[1], seq_lens[1], batch_size)
    test_loader = copying_data(sizes[2], seq_lens[2], batch_size)

    return train_loader, val_loader, test_loader


T = 80
n_train = 640
n_test = 640
n_sequence = 10
batch_size = 64

if __name__ == "__main__":
    train_x, train_y = copying_data(T, n_train, n_sequence,batch_size,make_rand=True)
    test_x, test_y = copying_data(T, n_test, n_sequence, batch_size,make_rand=True)

    print('train x y shapes', train_x.shape)
    print(train_y.shape)

    for j in range(0,2):
        print(train_x[0,j])
        print(train_y[0,j])

