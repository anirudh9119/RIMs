# coding: utf-8
import argparse
import datetime
import math
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from matplotlib.colors import LinearSegmentedColormap
from torchvision.utils import save_image

import baseline_lstm_model
import data
import mixed
import rnn_models
from cifar_seq_data_classify import cifar_data

# Set the random seed manually for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
np.random.seed(0)

def none_or_str(value):
    if value == 'None':
        return None
    return value

# same hyperparameter scheme as word-language-model
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=100,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--noise', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--tied', default=False, action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version. i.e. use RNN instead of RNNCell with for loop')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--resume', type=int, default=None,
                    help='if specified with the 1-indexed global epoch, loads the checkpoint and resumes training')
parser.add_argument('--algo', type=str, choices=('blocks', 'lstm','mixed'))
parser.add_argument('--num_blocks', nargs='+', type=int, default=[6])
parser.add_argument('--nhid', nargs='+', type=int, default=[300])
parser.add_argument('--topk', nargs='+', type=int, default=[4])
parser.add_argument('--block_dilation', nargs='+', type=int, default=-1)
parser.add_argument('--layer_dilation', nargs='+', type=int, default=-1)
parser.add_argument('--read_input', type=int, default=2)
parser.add_argument('--memory_slot', type=int, default=4)
parser.add_argument('--memory_heads', type=int, default=4)
parser.add_argument('--memory_head_size', type=int, default=16)
parser.add_argument('--gate_style', type=none_or_str, default=None)

parser.add_argument('--use_inactive', action='store_true',
                    help='Use inactive blocks for higher level representations too')
parser.add_argument('--blocked_grad', action='store_true',
                    help='Block Gradients through inactive blocks')
parser.add_argument('--scheduler', action='store_true',
                    help='Scheduler for Learning Rate')

# parameters for adaptive softmax
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')

# experiment name for this run
parser.add_argument('--name', type=str, default=None,
                    help='name for this experiment. generates folder with the name if specified.')

args = parser.parse_args()

best_val = {16: 0.0, 19:0.0, 24:0.0, 32:0.0}
best_test = {16: 0.0, 19:0.0, 24:0.0, 32:0.0}

best_val_epoch = {16:0.0, 19:0.0, 24:0.0, 32:0.0}
best_test_epoch = {16:0.0, 19:0.0, 24:0.0, 32:0.0}

#best_val = 0
#best_val_epoch = 0

inp_size = 16

######## Plot Specific Details ########

colors = ['white', 'black']
cmap = LinearSegmentedColormap.from_list('name', colors)
norm = plt.Normalize(0, 1)

matplotlib.rc('xtick', labelsize=7.5)
matplotlib.rc('ytick', labelsize=7.5)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

# Get Data Loaders

train_loader, val_loader, test_loader, noise_loader = cifar_data()

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

# create folder for current experiments
# name: args.name + current time
# includes: entire scripts for faithful reproduction, train & test logs
folder_name = str(datetime.datetime.now())[:-7]
if args.name is not None:
    folder_name = str(args.name)

if not os.path.exists(folder_name):
    os.mkdir(folder_name)
if not os.path.exists(folder_name+'/visuals/'):
    os.mkdir(folder_name+'/visuals/')


logger_args = open(os.path.join(os.getcwd(), folder_name, 'args.txt'), 'a')
logger_output = open(os.path.join(os.getcwd(), folder_name, 'output.txt'), 'a')
logger_epoch_output = open(os.path.join(os.getcwd(), folder_name, 'epoch_output.txt'), 'a')

# save args to logger
logger_args.write(str(args) + '\n')
savepath = os.path.join(os.getcwd(), folder_name)

###############################################################################
# Build the model
###############################################################################

ntokens = 256
n_out = 10

if args.adaptivesoftmax:
    print("Adaptive Softmax is on: the performance depends on cutoff values. check if the cutoff is properly set")
    print("Cutoffs: " + str(args.cutoffs))
    if args.cutoffs[-1] > ntokens:
        raise ValueError("the last element of cutoff list must be lower than vocab size of the dataset")
    criterion_adaptive = nn.AdaptiveLogSoftmaxWithLoss(args.nhid, ntokens, cutoffs=args.cutoffs).to(device)
else:
    criterion = nn.CrossEntropyLoss()


if args.algo == "blocks":
    rnn_mod = rnn_models.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, args.dropout, args.tied,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn,
                            use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs, use_inactive = args.use_inactive,
                            blocked_grad=args.blocked_grad, block_dilation=args.block_dilation,
                            layer_dilation=args.layer_dilation, num_modules_read_input=args.read_input).to(device)
elif args.algo == "lstm":
    rnn_mod = baseline_lstm_model.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, args.dropout, args.tied,
                            use_cudnn_version=args.cudnn,
                            use_adaptive_softmax=args.adaptivesoftmax, cutoffs=args.cutoffs).to(device)
elif args.algo == 'mixed':
    rnn_mod = mixed.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, args.dropout, args.tied,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs, use_inactive=args.use_inactive ,
                            blocked_grad=args.blocked_grad).to(device)
else:
    raise Exception("Algorithm option not found")

if os.path.exists(folder_name+'/model.pt'):
    state = torch.load(folder_name+'/model.pt')
    model.load_state_dict(state['state_dict'])
    global_epoch = state['epoch']
else:
    global_epoch = 1


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Built with Total Number of Trainable Parameters: " + str(total_params))
if not args.cudnn:
    print(
        "--cudnn is set to False. the model will use RNNCell with for loop, instead of cudnn-optimzed RNN API. Expect a minor slowdown.")

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

###############################################################################
# Load the model checkpoint if specified and restore the global & best epoch
###############################################################################

if args.resume is not None:
    print("--resume detected. loading checkpoint...")
global_epoch = args.resume if args.resume is not None else 0
best_epoch = args.resume if args.resume is not None else 0


if args.resume is not None:
    loadpath = os.path.join(os.getcwd(), "model_{}.pt".format(args.resume))
    if not os.path.isfile(loadpath):
        raise FileNotFoundError(
            "model_{}.pt not found. place the model checkpoint file to the current working directory.".format(
                args.resume))
    checkpoint = torch.load(loadpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_epoch = checkpoint["global_epoch"]
    best_epoch = checkpoint["best_epoch"]

print("Model Built with Total Number of Trainable Parameters: " + str(total_params))


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if args.algo == "lstm":
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)
    hidden = []
    for i in range(args.nlayers):
        if isinstance(h[i], torch.Tensor):
            hidden.append(h[i].detach())
        else:
            hidden.append(tuple((h[i][0].detach(), h[i][1].detach())))
    return hidden

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target

def mnist_prep(x, test_upsample=inp_size):
    # plt.imshow(x[0].transpose(2,0).transpose(0,1))
    # plt.show()
    d = x
    d = F.interpolate(d, size=(test_upsample,test_upsample), mode='nearest')
    d = d.transpose(3,1).transpose(1,2)
    d = d.reshape((d.shape[0],test_upsample*test_upsample,3)) * 255.
    d = d.round().to(dtype=torch.int64)
    # plt.imshow(d[0].reshape(test_upsample, test_upsample, 3))
    # plt.show()
    d = d.permute(1,0,2)
    return d

def evaluate_(test_lens, split):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    if split is "Val":
        loader = val_loader
    else:
        loader = test_loader

    test_acc = {i: 0.0 for i in test_lens}
    val_loss = 0.0

    for test_len in test_lens:
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for n, (d,t) in zip(noise_loader,loader):
                hidden = model.init_hidden(args.batch_size)

                if split is "Test" and args.noise:
                    d = torch.clamp(d + n/32., min=0.0, max=1.0)

                d = mnist_prep(d, test_upsample = test_len)
                t = t.to(dtype=torch.int64)

                data = d.cuda()
                targets = t.cuda()

                num_batches += 1

                output, hidden,extra_loss, _, _ = model(data, hidden)

                if not args.adaptivesoftmax:
                    loss = criterion(output[-1], targets)
                    acc = torch.eq(torch.max(output[-1],dim=1)[1], targets).double().mean()
                else:
                    _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)

                total_acc += acc.item()
                hidden = repackage_hidden(hidden)
                if test_len is inp_size:
                    val_loss += loss.item()

        test_acc[test_len] = total_acc / num_batches

    if split is "Val":
        val_loss = val_loss / num_batches
        if args.scheduler:
            scheduler.step(val_loss)

    return test_acc

def train():
    total_loss = 0.
    forward_elapsed_time = 0.
    start_time = time.time()

    i = 0
    j = 0

    calc_mask = False

    test_epoch = {16:0.0, 19:0.0, 24:0.0, 32:0.0}
    val_epoch = {16:0.0, 19:0.0, 24:0.0, 32:0.0}

    for d,t in train_loader:
        hidden = model.init_hidden(args.batch_size)
        model.train()
        i += 1

        d = mnist_prep(d, test_upsample=inp_size)
        t = t.to(dtype=torch.int64)

        device = torch.device("cuda:0")

        data =d.to(device)
        targets = t.to(device)

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        forward_start_time = time.time()

        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden, extra_loss, masks, sample_masks = model(data, hidden, calc_mask)

        if not args.adaptivesoftmax:
            loss = criterion(output[-1], targets)
            acc = torch.eq(torch.max(output[-1],dim=1)[1], targets).double().mean()
        else:
            raise Exception('not implemented')
            _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)
        total_loss += acc.item()

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed

        (loss + extra_loss).backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            printlog = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | forward ms/batch {:5.2f} | average acc {:5.4f} | ppl {:8.2f}'.format(
                epoch, i, len(train_loader.dataset) // args.batch_size, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, forward_elapsed_time * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss))
            # print and save the log
            print(printlog)
            logger_output.write(printlog + '\n')
            logger_output.flush()
            total_loss = 0.
            # reset timer
            start_time = time.time()
            forward_start_time = time.time()
            forward_elapsed_time = 0.

        if calc_mask and masks is not None:
            for k, (mask, sample_mask) in enumerate(zip(masks, sample_masks)):
                plt.imshow(mask.cpu().numpy(), cmap=cmap, norm=norm)
                plt.savefig(folder_name+'/visuals/'+'avg_layer{}_epoch{}.png'.format(str(k+1),str(epoch)))
                plt.close()
                plt.imshow(sample_mask.cpu().numpy(), cmap=cmap, norm=norm)
                plt.savefig(folder_name+'/visuals/'+'sample_layer{}_epoch{}.png'.format(str(k+1),str(epoch)))
                plt.close()
                for bl in range(args.num_blocks[k]):
                    plt.imshow(np.reshape(sample_mask.cpu().numpy()[bl,:], (inp_size,inp_size)), cmap=cmap, norm=norm)
                    plt.savefig(folder_name+'/visuals/'+'epoch{}_iter{}_sample_layer{}_block{}.png'.format(str(epoch), str(i), str(k+1), str(bl+1)))
                    plt.close()

            plt.imshow(d[:,0].view(inp_size,inp_size,3).cpu().numpy(), cmap=cmap, norm=norm)
            plt.savefig(folder_name+'/visuals/'+'epoch{}_iter{}.png'.format(str(epoch), str(i)))
            plt.close()
            if i > 5:
                calc_mask = False

    printlog = ''

# Loop over epochs.
best_val_loss = None

for epoch in range(1, args.epochs + 1):
    train()
    if epoch%4==0:
       test_lens = [16, 19,24,32]
       test_epoch = {16:0.0, 19:0.0, 24:0.0, 32:0.0}
       val_epoch = {16:0.0, 19:0.0, 24:0.0, 32:0.0}
       test_acc = evaluate_(test_lens, split="Test")
       val_acc = evaluate_(test_lens, split="Val")
       printlog = ''
       for key in test_acc:
          if val_acc[16] >  best_val[16]:
             print('Saving model', epoch)
             state = {
                'epoch': epoch,
                'state_dict': model.state_dict()
             }
             torch.save(state, folder_name+'/model.pt')

          if val_acc[key] > best_val[key]:
             best_val[key] = val_acc[key]
             best_test[key] = test_acc[key]

          test_epoch[key] += test_acc[key]
          val_epoch[key] += val_acc[key]

          printlog = printlog + '\n' + '|Seq_len: {} | Test Current: {} | Test Optim: {} | Val Current: {} | Val Best: {} |'.format(str(key), str(test_acc[key]), str(best_test[key]), str(val_acc[key]), str(best_val[key]))
       logger_output.write(printlog+'\n\n')
       logger_output.flush()
       print(printlog+'\n\n')
