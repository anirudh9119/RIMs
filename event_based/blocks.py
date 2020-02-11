
import torch
import torch.nn as nn

from attention import MultiHeadAttention
from BlockLSTM import BlockLSTM
from BlockGRU import BlockGRU
from sparse_grad_attn import blocked_grad

from blocks_core import BlocksCore

'''
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)

    output: 
    output, hx, cx

'''

class Blocks(nn.Module):

    def __init__(self, ninp, nhid, nlayers, num_blocks, top_k, use_inactive, blocked_grad, step_att=True, do_gru=False): 
        super(Blocks, self).__init__()
        self.nhid = nhid
        self.ninp = ninp
        self.top_k = top_k
        self.step_att = step_att
        self.do_gru = do_gru
        self.nlayers = nlayers
        self.num_blocks = num_blocks
        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad

        print("Number of Layers: ", nlayers)
        print("Input Dimension: ", ninp)
        print("Hidden Dimensions: ", nhid)
        print("Number of Blocks: ", num_blocks)
        print("Top k Blocks: ", top_k)
        print('Is the model using inactive blocks for higher representations? ', use_inactive)
        print('Is the model blocking gradients down inactive blocks? ', blocked_grad)

        self.bc_lst = []
        self.dropout_lst = []

        for i in range(nlayers):
            if i==0:
                self.bc_lst.append(BlocksCore(ninp, nhid[i], 1, num_blocks[i], top_k[i], True, do_gru=do_gru))
            else:
                self.bc_lst.append(BlocksCore(nhid[i-1], nhid[i], 1, num_blocks[i], top_k[i], True, do_gru=do_gru))

        self.bc_lst = nn.ModuleList(self.bc_lst)

    def blockify_params(self):
        for i in range(self.nlayers):
            self.bc_lst[i].block_lstm.blockify_params()

    def forward(self, inp, hx, cx, step):
        inp_use = inp

        hx_new, cx_new, mask_new = [],[],[]

        for idx_layer in range(self.nlayers):
            hx_, cx_, mask = self.bc_lst[idx_layer](inp_use, hx[idx_layer], cx[idx_layer], step)

            hx_new.append(hx_)
            cx_new.append(cx_)
            mask_new.append(mask)

            if self.use_inactive:
                if self.blocked_grad:
                    inp_use = blocked_grad.apply(hx_, mask)
                else:
                    inp_use = hx_
            else:
                if self.blocked_grad:
                    inp_use = blocked_grad.apply((mask)*hx_, mask)
                else:
                    inp_use = (mask)*hx_

        return hx_new, cx_new, mask_new


if __name__ == "__main__":
    bc = BlocksCore(512, 1, 4, 4)

    inp = torch.randn(10, 512)
    hx = torch.randn(10,512)
    cx = torch.randn(10,512)

    hx, cx = bc(inp, hx, cx)

    print('hx cx shape', hx.shape, cx.shape)