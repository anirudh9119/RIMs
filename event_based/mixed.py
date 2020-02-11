import torch.nn as nn
import torch
from attention import MultiHeadAttention
from layer_conn_attention import LayerConnAttention
from BlockLSTM import BlockLSTM
import random
import time
from GroupLinearLayer import GroupLinearLayer
from sparse_grad_attn import blocked_grad

from blocks_core import BlocksCore

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers=2, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, discrete_input=True, num_blocks=[6], topk=[4], do_gru=False,
                 use_inactive=False, lstm_layers=1, block_layers=1, blocked_grad=False):
        super(RNNModel, self).__init__()
        self.topk = topk
        print('Top k Blocks: ', topk)
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        print('Number of Inputs, ninp: ', ninp)
        if discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)
        self.num_blocks = num_blocks
        self.nhid = nhid
        print('Number of Blocks: ', self.num_blocks)
        self.discrete_input = discrete_input
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        nhid = nhid[0]
        print('Is the model using inactive blocks for higher representations? ', use_inactive)

        num_blocks_in = [1 for i in topk]
        self.lstm_layers = lstm_layers
        self.block_layers = block_layers

        self.bc_lst = []
        self.dropout_lst = []

        print("Dropout rate", dropout)

        for i in range(lstm_layers):
            self.bc_lst.append(getattr(nn,'LSTMCell')(ninp, nhid))

        for i in range(block_layers):
            self.bc_lst.append(BlocksCore(nhid, nhid, num_blocks_in[i], num_blocks[i], topk[i], True, do_gru=do_gru))

        for i in range(nlayers - 1):
            self.dropout_lst.append(nn.Dropout(dropout))

        self.bc_lst = nn.ModuleList(self.bc_lst)
        self.dropout_lst = nn.ModuleList(self.dropout_lst)

        if True:
            self.use_adaptive_softmax = use_adaptive_softmax
            self.decoder = nn.Linear(nhid, ntoken)
            if tie_weights:
                print('tying weights!')
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, calc_mask=False):

        extra_loss = 0.0

        emb = self.drop(self.encoder(input))

        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[], []]
            if calc_mask:
                masks = [[] for _ in range(self.block_layers)]
                sample_masks = [[] for _ in range(self.block_layers)]
            for idx_layer in range(self.nlayers):
                # print("idx layer: ", idx_layer)

                output = []
                t0 = time.time()
                
                #TODO: blockify
                if not idx_layer < self.lstm_layers:
                    self.bc_lst[idx_layer].blockify_params()
                #print('time to blockify', time.time() - t0)
                #print('hidden shape', hidden[0].shape)
                
                hx, cx = hidden[0][int(idx_layer)], hidden[1][int(idx_layer)]
                
                print_rand = random.uniform(0,1)
                
                for idx_step in range(input.shape[0]):
                    if idx_layer < self.lstm_layers:
                        hx, cx = self.bc_lst[idx_layer](layer_input[idx_step], (hx, cx))
                        mask = None
                    else:
                        hx, cx, mask = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx, idx_step, do_print=False)
                        mask = mask.detach()

                    if idx_layer < self.nlayers - 1 and idx_layer >= self.lstm_layers:
                        if self.use_inactive:
                            if self.blocked_grad:
                                bg = blocked_grad()
                                output.append(bg(hx,mask))
                            else:
                                output.append(hx)
                        else:
                            if self.blocked_grad:
                                bg = blocked_grad()
                                output.append((mask)*bg(hx,mask))
                            else:
                                output.append((mask)*hx)
                    else:
                        output.append(hx)

                    if calc_mask and mask is not None:
                        mask = mask.view(mask.size()[0], self.num_blocks[idx_layer-self.lstm_layers], self.nhid // self.num_blocks[idx_layer-self.lstm_layers])
                        mask = torch.mean(mask, dim=2)
                        sample_masks[idx_layer-self.lstm_layers].append(mask[0])
                        mask = torch.mean(mask, dim=0)
                        masks[idx_layer-self.lstm_layers].append(mask)

                if calc_mask and idx_layer >= self.lstm_layers:
                    masks[idx_layer-self.lstm_layers] = torch.stack(masks[idx_layer-self.lstm_layers]).view(input.shape[0],self.num_blocks[idx_layer-self.lstm_layers]).transpose(1,0)
                    sample_masks[idx_layer-self.lstm_layers] = torch.stack(sample_masks[idx_layer-self.lstm_layers]).view(input.shape[0],self.num_blocks[idx_layer-self.lstm_layers]).transpose(1,0)

                output = torch.stack(output)

                if idx_layer < self.nlayers - 1:
                    layer_input = self.dropout_lst[idx_layer](output)
                else:
                    layer_input = output
                
                new_hidden[0].append(hx)
                new_hidden[1].append(cx)

            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)

        output = self.drop(output)

        dec = output.view(output.size(0) * output.size(1), self.nhid)

        dec = self.decoder(dec)

        if calc_mask:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden, extra_loss, masks, sample_masks
        else:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden, extra_loss, None, None

    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].parameters())
        if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
