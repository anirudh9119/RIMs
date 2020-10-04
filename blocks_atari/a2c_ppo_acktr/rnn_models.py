import torch.nn as nn
import torch
from a2c_ppo_acktr.attention import MultiHeadAttention
from a2c_ppo_acktr.layer_conn_attention import LayerConnAttention
from a2c_ppo_acktr.BlockLSTM import BlockLSTM
import random
import time
from a2c_ppo_acktr.GroupLinearLayer import GroupLinearLayer
from a2c_ppo_acktr.sparse_grad_attn import blocked_grad

from a2c_ppo_acktr.blocks_core import BlocksCore

class Sparse_attention(nn.Module):
    def __init__(self, top_k = 5, num_rules=6):
        super(Sparse_attention,self).__init__()
        self.top_k = top_k
        self.num_rules = num_rules

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        #attn_s_max = torch.max(attn_s, dim = 1)[0]
        #attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[2]
        bottom_k = attn_s.size()[2] - self.top_k
        delta = torch.kthvalue(attn_s, bottom_k, dim= 2)[0]
        attn_w = attn_s - delta.repeat(1, self.num_rules).unsqueeze(1)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 2)
        attn_w_sum = attn_w_sum + eps 
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, self.num_rules).unsqueeze(1) 
        return attn_w_normalize


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, discrete_input=True, num_blocks=6, topk=4, do_gru=False,
                 num_modules_read_input=2):
        super(RNNModel, self).__init__()
        self.topk = topk
        print('top k blocks', topk)
        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(0.0)
        print('number of inputs, ninp', ninp)
        if discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)
        self.num_blocks = num_blocks
        self.nhid = nhid
        self.block_size = nhid // self.num_blocks
        print('number of blocks', self.num_blocks)
        self.discrete_input = discrete_input

        self.sigmoid = nn.Sigmoid()
        
        self.sm = nn.Softmax(dim=1)
        self.gate = GroupLinearLayer(self.block_size, 1, self.num_blocks)
        self.block_out = GroupLinearLayer(self.block_size,self.nhid,self.num_blocks)

        self.bc_lst = []

        print("Dropout rate", dropout)
        if True:
            self.bc_lst.append(BlocksCore(nhid, 1, num_blocks, topk, True, do_gru=do_gru, num_modules_read_input=num_modules_read_input))
            self.bc_lst = nn.ModuleList(self.bc_lst)

            if True:
                dropout_lst = []
                for i in range(nlayers):
                    dropout_lst.append(nn.Dropout(dropout))

                print('number of layers', nlayers)
                self.dropout_lst = nn.ModuleList(dropout_lst)
                print("Make dropout lst")
        if True:
            self.use_adaptive_softmax = use_adaptive_softmax
            self.decoder = nn.Linear(nhid, ntoken)
            if tie_weights:
                print('tying weights!')
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must be equal to emsize')
                self.decoder.weight = self.encoder.weight


        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.number_of_rules = 4#num_of_rules 
        self.num_of_steps_unrolling= 1#number_of_steps_unrolling
        self.output_ruleemb = 256 
        self.rule_emb = nn.Embedding(self.number_of_rules, self.output_ruleemb)

       
        self.sa = Sparse_attention(top_k=1, num_rules = self.number_of_rules)
        self.gate_fc = nn.Linear(self.block_size, self.block_size) 
        self.rule_block = MultiHeadAttention(n_head=1, d_model_read=self.block_size, d_model_write=self.output_ruleemb, 
                                             d_model_out=self.block_size, d_k=32, d_v=32, num_blocks_read=self.num_blocks, num_blocks_write=self.number_of_rules,
                                             residual=False, topk=self.number_of_rules, dropout = 0.1, skip_write=False, grad_sparse=False)
        
        self.rule_lst = []
        for i in range(self.number_of_rules):
            self.rules = nn.Sequential(
                    nn.Linear(self.block_size, 256), 
                    nn.Tanh(),
                    nn.Linear(256, self.block_size), nn.Tanh()) 
            self.rule_lst.append(self.rules)
        self.rule_lst = nn.ModuleList(self.rule_lst)
    
        self.num_gates = 2 #* self.calculate_gate_size()
        forget_bias = 1 
        input_bias = 0 
        self.input_gate_projector = nn.Linear(self.block_size, self.num_gates)
        self.memory_gate_projector = nn.Linear(self.block_size, self.num_gates)
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch.float32))

        self.init_weights()
       

    def create_gates(self, inputs, memory):
        """
        Create input and forget gates for this step using `inputs` and `memory`.
        Args:
          inputs: Tensor input.
          memory: The current state of memory.
        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        # We'll create the input and forget gates at once. Hence, calculate double
        # the gate size.
        #memory = torch.tanh(memory)

        # TODO: check this input flattening is correct
        # sonnet uses this, but i think it assumes time step of 1 for all cases
        # if inputs is (B, T, features) where T > 1, this gets incorrect
        # inputs = inputs.view(inputs.shape[0], -1)

        # fixed implementation
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError(
                    "input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1")
            inputs = inputs.view(inputs.shape[0], -1) 
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError("input shape of create_gate function is 2, expects 3")

        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]

        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)

        return input_gate, forget_gate 


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        extra_loss = 0.0
        timesteps, batch_size, _ = input.shape
        #emb = self.drop(self.encoder(input))
        emb = input
        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[], []]
            for idx_layer in range(0, self.nlayers):
                #print('idx layer', idx_layer)
                output = []
                masklst = []
                bmasklst = []
                t0 = time.time()
                #TODO: blockify
                self.bc_lst[idx_layer].blockify_params()
                #print('time to blockify', time.time() - t0)
                #print('hidden shape', hidden[0].shape)
                #hx, cx = hidden[0], hidden[1] #[idx_layer], hidden[1][idx_layer]
                hx, cx = hidden[0][idx_layer], hidden[1][idx_layer]
                do_print = False
                for idx_step in range(input.shape[0]):
                    hx, cx, mask, bmask = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx, idx_step, do_print=do_print)
                    output.append(hx)
                    masklst.append(mask)
                    bmasklst.append(bmask)

                output = torch.stack(output)
                mask = torch.stack(masklst)
                layer_input = output
                new_hidden[0].append(hx)
                new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)

        block_mask = bmask.squeeze(0)
        assert input.shape[1] == hx.shape[0]
        ##print("Going to rules")
        ### Step 3: Write to blocks.
        '''
        for num_ in range(self.num_of_steps_unrolling):
            hx, cx = hidden[0], hidden[1]
            #TODO: This detach function is justified as this is being used for selection of rules and blocks.
            block_repr = hx.view(batch_size, self.num_blocks, self.block_size).detach()
            # block_repr = hx.reshape(batch_size * self.num_blocks, self.block_size).detach()
            block_repr = block_repr.unsqueeze(1).repeat(1, self.number_of_rules, 1, 1)
            # batch, rule, block, dim
            block_mask = bmask.squeeze(0)
            input_em = (torch.ones(1, self.number_of_rules).cumsum(dim=1) - 1).type(torch.LongTensor).cuda()
 
            rule_repr2 = self.rule_emb(input_em)
            #rule_repr2 = rule_repr2.unsqueeze(0).repeat(batch_size, 1, 1) 

            if batch_size > 1:
                rule_repr2 = rule_repr2.repeat(batch_size, 1, 1)
                
 
            something_6, iatt_6, _ = self.rule_block(hx.squeeze(0).reshape(batch_size, self.num_blocks, self.block_size).clone().detach(),
                                                    rule_repr2.reshape(batch_size, self.number_of_rules, self.output_ruleemb),
                                                    rule_repr2.reshape(batch_size,self.number_of_rules, self.output_ruleemb))
            
            bmk_= bmask.squeeze(0).squeeze(2).unsqueeze(1).repeat(1,self.number_of_rules, 1)
            iatt_6  = iatt_6.transpose(1,2)
            iatt_ = torch.mul(iatt_6, bmk_)

            #print(iatt_[23])
            output = []
            hx = hx.squeeze(0)
            cx = cx.squeeze(0)
            output = [] 
            for i in range(self.num_blocks):
                start_index, end_index = i * self.block_size, (i + 1) * self.block_size
                memory = hx[:, start_index: end_index]
                te_ = []
                # TODO: Ideally I've detached here, but it should not be detached.
                for j_ in range(self.number_of_rules):
                    temp_out = self.drop2(
                        self.rule_lst[j_](memory).detach())
                    te_.append(temp_out)
                qw_ = self.sa(iatt_[:, :, i].unsqueeze(dim=2).transpose(1, 2))
                te_ = torch.stack(te_, dim=1).requires_grad_(True)
                result_ = te_ * qw_.squeeze(dim=1).unsqueeze(dim=2)
                result_ = result_.sum(dim=1)
                input_gate, forget_gate = self.create_gates(inputs= result_.unsqueeze(1),
                                                            memory=memory.unsqueeze(1))

                next_memory = memory + input_gate.squeeze(2) * ( torch.tanh(result_) - memory)
                hx[:, start_index: end_index] = next_memory
            
            hx, cx, extra_temp_loss = self.bc_lst[0].step_attention(hx, cx, masklst[0])
            extra_loss  += extra_temp_loss
            new_hidden = [[], []]
            output.append(hx)
            output = torch.stack(output).requires_grad_(True)
            new_hidden[0].append(hx)
            new_hidden[1].append(cx)
            new_hidden[0] = torch.stack(new_hidden[0])
            new_hidden[1] = torch.stack(new_hidden[1])
            hidden = tuple(new_hidden)
        '''
        #print("Going outside rules")
        output = self.drop(output)
        dec = output.view(output.size(0) * output.size(1), self.nhid)
        dec = self.decoder(dec)
        return dec.view(output.size(0), output.size(1), dec.size(1)), hidden#, extra_loss

        #if not self.use_adaptive_softmax:
            #print('not use adaptive softmax, size going into decoder', output.size())
        #    decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        #    return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, extra_loss

    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].block_lstm.parameters())
        if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
