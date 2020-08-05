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
import data
import argparse
import time
import math
import os
import torch.nn as nn
import torch.onnx
import datetime
import shutil
import pickle
import rnn_models_wiki#adding
import baseline_lstm_model_adding
import random
import mixed
#from adding_data import adding_dataset
from copying_data import copying_data
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

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
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
parser.add_argument('--epochs', type=int, default=200,
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
parser.add_argument('--train_len', type=int, default=500)
parser.add_argument('--test_len', type=int, default=1000)
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
parser.add_argument('--adaptivesoftmax', action='store_true',
                    help='use adaptive softmax during hidden state to output logits.'
                         'it uses less memory by approximating softmax of large vocabulary.')
parser.add_argument('--cutoffs', nargs="*", type=int, default=[10000, 50000, 100000],
                    help='cutoff values for adaptive softmax. list of integers.'
                         'optimal values are based on word frequencey and vocabulary size of the dataset.')
parser.add_argument('--name', type=str, default=None,
                    help='name for this experiment. generates folder with the name if specified.')

args = parser.parse_args()

print("Tied?", args.tied)

# Set the random seed manually for reproducibility.

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")



eval_batch_size = 32


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

ntokens = 20#len(corpus.dictionary)
#print("vocabulary size (ntokens): " + str(ntokens))
if args.adaptivesoftmax:
    print("Adaptive Softmax is on: the performance depends on cutoff values. check if the cutoff is properly set")
    print("Cutoffs: " + str(args.cutoffs))
    if args.cutoffs[-1] > ntokens:
        raise ValueError("the last element of cutoff list must be lower than vocab size of the dataset")
    criterion_adaptive = nn.AdaptiveLogSoftmaxWithLoss(args.nhid, ntokens, cutoffs=args.cutoffs).to(device)
else:
    criterion = nn.CrossEntropyLoss()

if args.algo == "blocks":
    rnn_mod = rnn_models_wiki.RNNModel#.RNNModel
    model = rnn_mod(args.model, ntokens, args.emsize, args.nhid,
                            args.nlayers, args.dropout, args.tied,
                            num_blocks = args.num_blocks, topk = args.topk,
                            use_cudnn_version=args.cudnn, use_adaptive_softmax=args.adaptivesoftmax,
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


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("model built, total trainable params: " + str(total_params))
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

print("model built, total trainable params: " + str(total_params))


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


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


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = 20#len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    print('eval data source shape', data_source.shape)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden,extra_loss = model(data, hidden)
            if not args.adaptivesoftmax:
                loss = criterion(output.view(-1, ntokens), targets)
            else:
                _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)
            total_loss += len(data) * loss#.item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def evaluate_(copy_x, copy_y):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = 20#len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    num_batches = 200
    calc_mask=False
    with torch.no_grad():
        for i in range(num_batches):
            batch_ind = random.randint(0, num_batches-1)
            data = Variable(copy_x[batch_ind].cuda()) if args.cuda else Variable(copy_x[batch_ind])
            targets = Variable(copy_y[batch_ind].cuda()) if args.cuda else Variable(copy_y[batch_ind])
            #output, hidden,extra_loss = model(data, hidden)
            output, hidden, extra_loss, _, _ = model(data, hidden, calc_mask)
            if not args.adaptivesoftmax:
                loss = criterion(output.view(-1, ntokens), targets.reshape((args.test_len + 20) * 64))
            else:
                raise Exception('not implemented')
                _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)
            total_loss += loss.item()
            hidden = repackage_hidden(hidden)
    return total_loss/200 #/ len(data_source)





def train(epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    forward_elapsed_time = 0.
    start_time = time.time()
    ntokens = 20#len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)

    copy_x, copy_y = copying_data(T=args.train_len)
    eval_copy_x, eval_copy_y = copying_data(T=args.test_len)
    num_batches = 200
    calc_mask = False
    for i in range(num_batches):
        batch_ind = random.randint(0, num_batches-1)

        #data, targets = get_batch(train_data, i)
        data = Variable(copy_x[batch_ind].cuda()) if args.cuda else Variable(copy_x[batch_ind])
        targets = Variable(copy_y[batch_ind].cuda()) if args.cuda else Variable(copy_y[batch_ind])

        if args.cuda:
            torch.cuda.synchronize()
        forward_start_time = time.time()
        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden, extra_loss, masks, sample_masks = model(data, hidden, calc_mask)
        if not args.adaptivesoftmax:
            loss = criterion(output.view(-1, ntokens), targets.reshape((args.train_len + 20) * 64))
        else:
            raise Exception('not implemented')
            _, loss = criterion_adaptive(output.view(-1, args.nhid), targets)
        total_loss += loss.item()

        if args.cuda:
            torch.cuda.synchronize()

        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed

        (loss + extra_loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            printlog = '| epoch {:3d} | {:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | forward ms/batch {:5.2f} | loss {:5.4f} | ppl {:8.2f}'.format(
                epoch,i , optimizer.param_groups[0]['lr'],
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


        if i % args.log_interval == 0 and i > 0:
            test_loss = evaluate_(eval_copy_x, eval_copy_y)
            printlog = '\n' + '| Test Current: {} '.format(str(test_loss))
            #print(printlog)

            logger_output.write(printlog+'\n\n')
            logger_output.flush()

            #print(printlog+'\n\n')
            print("test loss is %d", test_loss)

for epoch in range(global_epoch + 1, args.epochs + 1):
    train(epoch)
