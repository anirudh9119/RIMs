
import rnn_models
import torch
import torch.nn as nn

#A wrapper for using Blocks model.
class BlocksWrapper(nn.Module):

    def __init__(self, ntokens, nhid, n_out, dropout=0.0, num_blocks=4, update_topk=4):
        super(BlocksWrapper, self).__init__()
        self.myrnn = rnn_models.RNNModel("LSTM", ntokens, nhid, nhid,
                            nlayers=1, dropout=dropout, tie_weights=False,
                            use_cudnn_version=False, use_adaptive_softmax=False,
                            cutoffs=[10000], discrete_input=False, n_out = n_out, num_blocks=num_blocks, update_topk=update_topk).cuda()
        #self.myrnn = nn.LSTM(ntokens, nhid)
        self.nhid = nhid

        print('using blocks wrapper!')

    def forward(self, inp, h):
        hx = h[:,:,:self.nhid].contiguous()
        cx = h[:,:,self.nhid:].contiguous()
        ob, (hx,cx) = self.myrnn(inp, (hx, cx))
        hb = torch.cat([hx,cx], dim=2)
        return ob,hb


if __name__ == "__main__":
    nhid = 128
    ntokens = 128

    blocks = BlocksWrapper(ntokens, nhid, n_out=nhid).cuda()
    gru = torch.nn.GRU(ntokens, nhid).cuda()

    x = torch.randn(1, 1, ntokens).cuda()

    h0 = torch.randn(1, 1, nhid).cuda()
    h0_blocks = torch.randn(1, 1, nhid*2).cuda()

    og, hg = gru(x, h0)
    print('gru of x: o,h', og.shape, hg.shape)

    ob, hb = blocks(x, h0_blocks)
    print('block res: o,h', ob.shape, hb.shape)



