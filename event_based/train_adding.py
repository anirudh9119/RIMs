# coding: utf-8
import numpy as np
import torch
import random

# Set the random seed manually for reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
np.random.seed(0)

import argparse
import time
import math
import os
import torch.nn as nn
import torch.onnx
import datetime
import shutil
import pickle
import rnn_models_adding
import baseline_lstm_model_adding
import random
import mixed
from adding_data import adding_dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib


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
parser.add_argument('--algo', type=str, choices=('blocks', 'lstm'))
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
parser.add_argument('--train_len', type=int, default=500)
parser.add_argument('--test_len', type=int, default=1000)
parser.add_argument('--use_inactive', action='store_true',
                    help='Use inactive blocks for higher level representations too')
parser.add_argument('--blocked_grad', action='store_true',
                    help='Block Gradients through inactive blocks')
parser.add_argument('--scheduler', action='store_true',
                    help='Scheduler for Learning Rate')
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')
parser.add_argument('--name', type=str, default=None,
                    help='name for this experiment. generates folder with the name if specified.')

args = parser.parse_args()

best_val = math.inf
best_test = math.inf

best_val_epoch = math.inf
best_test_epoch = math.inf

sizes = [30000,10000,20000]
lens = [args.train_len, args.test_len, args.test_len]

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
train_loader, val_loader, test_loader = adding_dataset(sizes, lens, args.batch_size)

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

logger_args = open(os.path.join(os.getcwd(), folder_name, 'args.txt'), 'w+')
logger_output = open(os.path.join(os.getcwd(), folder_name, 'output.txt'), 'w+')
logger_epoch_output = open(os.path.join(os.getcwd(), folder_name, 'epoch_output.txt'), 'w+')

# save args to logger
logger_args.write(str(args) + '\n')

# define saved model file location
savepath = os.path.join(os.getcwd(), folder_name)

###############################################################################
# Build the model
###############################################################################

inp_dim = 3
out_dim = 1

criterion = nn.MSELoss()


if args.algo == "blocks":
    rnn_mod = rnn_models_adding.RNNModel
    model = rnn_mod(args.model, inp_dim, out_dim, args.emsize, args.nhid,
                            args.nlayers, args.dropout, args.tied,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn,
                            use_adaptive_softmax=args.adaptivesoftmax,
                            cutoffs=args.cutoffs, use_inactive = args.use_inactive,
                            blocked_grad=args.blocked_grad, block_dilation=args.block_dilation,
                            layer_dilation=args.layer_dilation, num_modules_read_input=args.read_input).to(device)
elif args.algo == "lstm":
    rnn_mod = baseline_lstm_model_adding.RNNModel
    model = rnn_mod(args.model, inp_dim, out_dim, args.emsize, args.nhid,
                            args.nlayers, args.dropout, args.tied,
                            use_cudnn_version=args.cudnn,
                            use_adaptive_softmax=args.adaptivesoftmax, cutoffs=args.cutoffs).to(device)
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

def evaluate(split):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    calc_mask=False
    if split is "Val":
        data_source = val_loader
    else:
        data_source = test_loader

    total_loss = 0.0

    with torch.no_grad():
        for d,t in data_source:
            hidden = model.init_hidden(d.shape[0])

            d = d.transpose(1,0)

            if args.cuda:
                data = Variable(d.cuda())
                targets = Variable(t.cuda())
            else:
                data = Variable(d)
                targets=Variable(t)

            output, hidden, extra_loss, _, _ = model(data, hidden, calc_mask)
            pred = output[-1].reshape(targets.shape[0])

            loss = criterion(pred, targets)

            total_loss += len(data) * loss.item()

    total_loss /= len(data_source.dataset)

    if split is "Val":
        if args.scheduler:
            scheduler.step(total_loss)

    return total_loss

def train(epoch):
    global best_val, best_test, best_val_epoch, best_test_epoch
    total_loss = 0.
    forward_elapsed_time = 0.
    start_time = time.time()

    i = 0
    j = 0

    calc_mask = False

    test_epoch = 0.0
    val_epoch = 0.0

    for batch,(d,t) in enumerate(train_loader):
        hidden = model.init_hidden(args.batch_size)
        model.train()

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        forward_start_time = time.time()
        model.zero_grad()

        d = d.transpose(1,0)

        if args.cuda:
            data = Variable(d.cuda())
            targets = Variable(t.cuda())
        else:
            data = Variable(d)
            targets=Variable(t)

        output, hidden, extra_loss, masks, sample_masks = model(data, hidden, calc_mask)
        pred = output[-1].reshape(targets.shape[0])

        loss = criterion(pred, targets)

        total_loss += loss.item()

        # synchronize cuda for a proper speed benchmark
        torch.cuda.synchronize()

        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            printlog = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | forward ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_loader.dataset) // args.batch_size, optimizer.param_groups[0]['lr'],
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

        if batch % args.log_interval == 0 and batch > 0 and epoch % 5 == 0:
            j += 1
            test_loss = evaluate(split="Test")
            val_loss = evaluate(split="Val")

            printlog = ''

            if val_loss < best_val:
                best_val = val_loss
                best_test = test_loss
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict()
                    }
                torch.save(state, folder_name+'/best_model.pt')


            test_epoch += test_loss
            val_epoch += val_loss

            printlog = printlog + '\n' + '| Test Current: {} | Test Optim: {} | Val Current: {} | Val Best: {} |'.format(str(test_loss), str(best_test), str(val_loss), str(best_val))

            logger_output.write(printlog+'\n\n')
            logger_output.flush()

            print(printlog+'\n\n')

    printlog = ''

    try:
        avg_test = test_epoch / j
        avg_val = val_epoch / j

        if avg_val < best_val_epoch:
            best_val_epoch = avg_val
            best_test_epoch = avg_test

        printlog = printlog + '\n' + '| Test: {} | Optimum: {} | Val: {} | Best Val: {} |'.format(str(avg_test), str(best_test_epoch), str(avg_val), str(best_val_epoch))

        logger_epoch_output.write(printlog+'\n\n')
        logger_epoch_output.flush()
        print(printlog+'\n\n')
    except:
        pass

    state = {
    'epoch': epoch,
    'state_dict': model.state_dict()
    }
    torch.save(state, folder_name+'/model.pt')

for epoch in range(global_epoch, args.epochs + 1):
    train(epoch)
