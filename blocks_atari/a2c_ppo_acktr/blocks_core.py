
import torch
import torch.nn as nn

from a2c_ppo_acktr.attention import MultiHeadAttention
from a2c_ppo_acktr.BlockLSTM import BlockLSTM
from a2c_ppo_acktr.BlockGRU import BlockGRU
from a2c_ppo_acktr.sparse_grad_attn import blocked_grad
'''
Core blocks module.  Takes:
    input: (ts, mb, h)
    hx: (ts, mb, h)
    cx: (ts, mb, h)

    output:
    output, hx, cx

'''
from torch.distributions.categorical import Categorical

class BlocksCore(nn.Module):


    def __init__(self, nhid, num_blocks_in, num_blocks_out, topkval, step_att, do_gru, num_modules_read_input=2):
        super(BlocksCore, self).__init__()
        self.nhid = nhid
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.block_size_in = nhid // num_blocks_in
        self.block_size_out = nhid // num_blocks_out
        self.topkval = topkval
        self.step_att = step_att
        self.do_gru = do_gru
        self.num_modules_read_input = num_modules_read_input

        print('bs in', self.block_size_in)
        print('bs out', self.block_size_out)
        print('topk, num_modules_read_input', self.topkval, self.num_modules_read_input)

        self.mha = MultiHeadAttention(n_head=4, d_model_read=self.block_size_out, d_model_write=self.block_size_out, d_model_out=self.block_size_out, d_k=32, d_v=32, num_blocks_read=self.num_blocks_out, num_blocks_write=self.num_blocks_out, topk=self.num_blocks_out, grad_sparse=False)


        self.version = 2 #version
        if self.version == 1:
            self.att_out = self.block_size_out*4
            print('Using version 1 att_out is', self.att_out)
            self.inp_att = MultiHeadAttention(n_head=1, d_model_read=self.block_size_out, d_model_write=self.block_size_in,
                                              d_model_out=self.att_out, d_k=64, d_v=self.att_out, num_blocks_read=num_blocks_out,
                                              num_blocks_write=num_modules_read_input, residual=False, topk=self.num_blocks_in+1,
                                              grad_sparse=False, skip_write=True)


        elif self.version == 2:
            self.att_out = self.block_size_out*4
            print('Using version 2 att_out is', self.att_out)
            d_v = self.att_out#//self.inp_heads
            self.inp_att = MultiHeadAttention(n_head=1, d_model_read=self.block_size_out,
                                           d_model_write=self.block_size_in, d_model_out=self.att_out,
                                           d_k=64, d_v=self.att_out, num_blocks_read=num_blocks_out, num_blocks_write=1, residual=False,
                                           dropout=0.1, topk=topkval, grad_sparse=False, skip_write=True, flag=True)

        if do_gru:
            self.block_lstm = BlockGRU(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)
        else:
            self.block_lstm = BlockLSTM(self.att_out*self.num_blocks_out, self.nhid, k=self.num_blocks_out)

    def blockify_params(self):
        self.block_lstm.blockify_params()

    def forward(self, inp, hx, cx, step,do_print=False):

        hxl = []
        cxl = []
        inp_use = inp #layer_input[idx_step]

        if self.version == 1:
             inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks_in, self.block_size_in))
             inp_use = inp_use.repeat(1,self.num_modules_read_input-1,1)
             inp_use = torch.cat([torch.zeros_like(inp_use[:,0:1,:]), inp_use], dim=1)


             inp_use, iatt, _ = self.inp_att(hx.reshape((hx.shape[0], self.num_blocks_out, self.block_size_out)), inp_use, inp_use)
             inp_use = inp_use.reshape((inp_use.shape[0], self.att_out*self.num_blocks_out))

             new_mask = torch.ones_like(iatt[:,:,0])
             if (self.num_blocks_out - self.topkval)>0:
                 bottomk_indices = torch.topk(iatt[:,:,0], dim=1,
                                         sorted=True, largest=True,
                                         k = self.num_blocks_out - self.topkval)[1]

                 new_mask.index_put_((torch.arange(bottomk_indices.size(0)).unsqueeze(1), bottomk_indices),
                     torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype))

        if self.version == 2:
            inp_use = inp_use.reshape((inp_use.shape[0], self.num_blocks_in, self.block_size_in))
            batch_size = inp.shape[0]
            inp_use, iatt, _ = self.inp_att(hx.reshape((hx.shape[0], self.num_blocks_out, self.block_size_out)), inp_use, inp_use)
            iatt = iatt.reshape((1, inp_use.shape[0], iatt.shape[1], iatt.shape[2]))
            iatt = iatt.mean(0)
            self.iatt_log = iatt.clone().detach().cpu().numpy()
            inp_use = inp_use.reshape((inp_use.shape[0], self.att_out*self.num_blocks_out))

            new_mask = torch.ones_like(iatt[:,:,0])
            if (self.num_blocks_out - self.topkval)>0:
                bottomk_indices = torch.topk(iatt[:,:,0], dim=1,
                                  sorted=True, largest=False,
                                  k = self.num_blocks_out - self.topkval)[1]
                new_mask.index_put_((torch.arange(bottomk_indices.size(0)).unsqueeze(1), bottomk_indices),
                     torch.zeros_like(bottomk_indices[0], dtype=new_mask.dtype))


        mask = new_mask
        block_mask = mask.reshape((inp_use.shape[0], self.num_blocks_out,1))

        if do_print:
            print('mask at', step, mask[0])


        mask = mask.reshape((inp_use.shape[0],self.num_blocks_out,1)).repeat((1,1,self.block_size_out)).reshape((inp_use.shape[0], self.num_blocks_out*self.block_size_out))

        mask = mask.detach()

        if self.do_gru:
            hx_new = self.block_lstm(inp_use, hx)
            cx_new = hx_new
        else:
            hx_new, cx_new = self.block_lstm(inp_use, hx, cx)

        hx_old = hx*1.0
        cx_old = cx*1.0

        if self.step_att:
            hx_new = hx_new.reshape((hx_new.shape[0], self.num_blocks_out, self.block_size_out))
            #bg = blocked_grad()
            hx_new_grad_mask = blocked_grad.apply(hx_new,
                                                  mask.reshape(
                                                      (mask.shape[0],
                                                       self.num_blocks_out,
                                                       self.block_size_out)))
            hx_new_att,attn_out,extra_loss_att = self.mha(hx_new_grad_mask,hx_new_grad_mask,hx_new_grad_mask)
            hx_new = hx_new + hx_new_att
            hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
            extra_loss = extra_loss_att

        hx = (mask)*hx_new + (1-mask)*hx_old
        cx = (mask)*cx_new + (1-mask)*cx_old

        return hx, cx, mask, block_mask

    def step_attention(self, hx_new, cx_new, mask):
        hx_new = hx_new.reshape((hx_new.shape[0], self.num_blocks_out, self.block_size_out))
        # bg = blocked_grad()
        hx_new_grad_mask = blocked_grad.apply(hx_new,
                                              mask.reshape((mask.shape[0],
                                                            self.num_blocks_out,
                                                            self.block_size_out)))
        hx_new_att,attn_out,extra_loss_att = self.mha(hx_new_grad_mask,hx_new_grad_mask,hx_new_grad_mask)
        hx_new = hx_new + hx_new_att
        hx_new = hx_new.reshape((hx_new.shape[0], self.nhid))
        extra_loss = extra_loss_att
        return hx_new, cx_new, extra_loss

if __name__ == "__main__":
    bc = BlocksCore(512, 1, 4, 4)

    inp = torch.randn(10, 512)
    hx = torch.randn(10,512)
    cx = torch.randn(10,512)

    hx, cx = bc(inp, hx, cx)

    print('hx cx shape', hx.shape, cx.shape)





