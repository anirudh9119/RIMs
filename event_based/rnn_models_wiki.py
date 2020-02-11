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

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, discrete_input=True, num_blocks=[6], topk=[4], do_gru=False,
                 use_inactive=False, blocked_grad=False, layer_dilation = -1, block_dilation = -1, num_modules_read_input=2):

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
        print('Number of Blocks: ', self.num_blocks)

        self.nhid = nhid
        print('Dimensions of Hidden Layers: ', nhid)

        self.discrete_input = discrete_input
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)

        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        print('Is the model using inactive blocks for higher representations? ', use_inactive)

        if layer_dilation == -1:
            self.layer_dilation = [1]*nlayers
        else:
            self.layer_dilation = layer_dilation

        if block_dilation == -1:
            self.block_dilation = [1]*nlayers
        else:
            self.block_dilation = block_dilation

        num_blocks_in = [1 for i in topk]

        self.bc_lst = []
        self.dropout_lst = []

        print("Dropout rate", dropout)

        for i in range(nlayers):
            if i==0:
                self.bc_lst.append(BlocksCore(ninp,nhid[i], num_blocks_in[i], num_blocks[i], topk[i], True, do_gru=do_gru, num_modules_read_input=num_modules_read_input))
            else:
                self.bc_lst.append(BlocksCore(nhid[i-1],nhid[i], num_blocks_in[i], num_blocks[i], topk[i], True, do_gru=do_gru, num_modules_read_input=num_modules_read_input))
        for i in range(nlayers - 1):
            self.dropout_lst.append(nn.Dropout(dropout))

        self.bc_lst = nn.ModuleList(self.bc_lst)
        self.dropout_lst = nn.ModuleList(self.dropout_lst)

        if True:
            self.use_adaptive_softmax = use_adaptive_softmax
            self.decoder = nn.Linear(nhid[-1], ntoken)
            if tie_weights:
            	if nhid[-1] != ninp:
            		raise ValueError('When using the tied flag, nhid must be equal to emsize')
            	else:
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
            new_hidden = [[] for _ in range(self.nlayers)]
            if calc_mask:
                masks = [[] for _ in range(self.nlayers)]
                sample_masks = [[] for _ in range(self.nlayers)]
            for idx_layer in range(self.nlayers):
                output = []
                t0 = time.time()
                self.bc_lst[idx_layer].blockify_params()
                hx, cx = hidden[int(idx_layer)][0], hidden[int(idx_layer)][1]
                for idx_step in range(input.shape[0]):
                    if idx_step % self.layer_dilation[idx_layer] == 0:
                        if idx_step % self.block_dilation[idx_layer] == 0:
                            hx, cx, mask = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx, idx_step, do_block = True)
                        else:
                            hx, cx, mask = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx, idx_step, do_block = False)

                    if idx_layer < self.nlayers - 1:
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

                    if calc_mask:
                        mk = mask.view(mask.size()[0], self.num_blocks[idx_layer], self.nhid[idx_layer] // self.num_blocks[idx_layer])
                        mk = torch.mean(mk, dim=2)
                        sample_masks[idx_layer].append(mk[0])
                        mk = torch.mean(mk, dim=0)
                        masks[idx_layer].append(mk)

                if calc_mask:
                    masks[idx_layer] = torch.stack(masks[idx_layer]).view(input.shape[0],self.num_blocks[idx_layer]).transpose(1,0)
                    sample_masks[idx_layer] = torch.stack(sample_masks[idx_layer]).view(input.shape[0],self.num_blocks[idx_layer]).transpose(1,0)

                output = torch.stack(output)

                if idx_layer < self.nlayers - 1:
                    layer_input = self.dropout_lst[idx_layer](output)
                else:
                    layer_input = output

                new_hidden[idx_layer] = tuple((hx,cx))

            hidden = new_hidden

        output = self.drop(output)
        dec = output.view(output.size(0) * output.size(1), self.nhid[-1])
        dec = self.decoder(dec)
        if calc_mask:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden, extra_loss, masks, sample_masks
        else:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden, extra_loss, None, None


    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].block_lstm.parameters())
        hidden = []
        if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            for i in range(self.nlayers):
                hidden.append((weight.new_zeros(bsz, self.nhid[i]),
                    weight.new_zeros(bsz, self.nhid[i])))
            # return (weight.new_zeros(self.nlayers, bsz, self.nhid),
            #         weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            for i in range(self.nlayers):
                hidden.append((weight.new_zeros(bsz, self.nhid[i])))

        return hidden
