#========================================================
#             GRUCell.py - implementation of Gated Recurrent Unit
#========================================================

import math
import torch
import torch.nn as nn


# ****************************************
# TODO: implement a complete GRUCell class, please refer to the implementation of RNNCell in RNNCell.py
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()

        # TODO: define all learnable parameters and activation functions here
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ir = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hr = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_iz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_in = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.weight_hn = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        self.bias_ir = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hr = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_iz = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hz = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_in = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hn = nn.Parameter(torch.Tensor(hidden_size))
        
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, input, hidden):
        '''
        :param input: input with size [batch_size, input_size]
        :param hidden: hidden state at previous time step, size [batch_size, hidden_size]
        :return: hidden state with size [batch_size, hidden_size]
        '''

        r_t = self.sigmoid(torch.matmul(input, self.weight_ir) + self.bias_ir + torch.matmul(hidden, self.weight_hr) + self.bias_hr)
        z_t = self.sigmoid(torch.matmul(input, self.weight_iz) + self.bias_iz + torch.matmul(hidden, self.weight_hz) + self.bias_hz)
        h_tbar = self.Tanh(torch.matmul(input, self.weight_in) + self.bias_in + r_t * (torch.matmul(hidden, self.weight_hn) + self.bias_hn))
        h_t = (1 - z_t) * h_tbar + z_t * hidden
        return h_t

if __name__ == '__main__':
    # define the GRU model
    gru = GRUCell(input_size=2, hidden_size=3)

    with torch.no_grad():
        # set model's parameter
        gru.weight_ir.copy_(torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float))
        gru.bias_ir.copy_(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float))
        gru.weight_hr.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float))
        gru.bias_hr.copy_(torch.tensor([-0.1, -0.2, 0.3], dtype=torch.float))

        gru.weight_iz.copy_(torch.tensor([[0.0, 0.8, 0.6], [-0.5, 1.0, 0.0]], dtype=torch.float))
        gru.bias_iz.copy_(torch.tensor([0.3, 0.2, 0.1], dtype=torch.float))
        gru.weight_hz.copy_(torch.tensor([[2.0, 1.0, 1.0], [0.8, 1.2, -0.7], [-0.3, 0.8, -1.0]], dtype=torch.float))
        gru.bias_hz.copy_(torch.tensor([-0.2, -0.2, 0.2], dtype=torch.float))

        gru.weight_in.copy_(torch.tensor([[1.0, -1.0, 0.0], [0.6, -0.3, -0.4]], dtype=torch.float))
        gru.bias_in.copy_(torch.tensor([-0.5, -0.5, -0.5], dtype=torch.float))
        gru.weight_hn.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float))
        gru.bias_hn.copy_(torch.tensor([-0.1, 0.7, 0.2], dtype=torch.float))


    h0 = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float)
    x1 = torch.tensor([[0.1, -0.2], [-0.7, -0.2]], dtype=torch.float)
    x2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float)

    # process for model forwarding 
    h1 = gru(x1, h0)
    h2 = gru(x2, h1)
 
    print('[Info] testing GRUCell...')
    h1_true = torch.load('data/GRUCell_results/h1.pth')
    h2_true = torch.load('data/GRUCell_results/h2.pth')
    if not h1.shape == h1_true.shape:
        print('[Error] shape error, your shapes do not match the expected shape.')
        print('Wrong shape for h1')
        print('Your shape:    ', h1.shape)
        print('Expected shape:', h1_true.shape)
        exit(1)
    if not torch.allclose(h1, h1_true, atol=1e-05):
        print('[Error] closeness error, your values do not match the expected values.')
        print('Wrong values for h1')
        print('Your values:    ', h1)
        print('Expected values:', h1_true)
        exit(1)
    
    if not h2.shape == h2_true.shape:
        print('[Error] shape error, your shapes do not match the expected shape.')
        print('Wrong shape for h2')
        print('Your shape:    ', h2.shape)
        print('Expected shape:', h2_true.shape)
        exit(1)
    if not torch.allclose(h2, h2_true, atol=1e-05):
        print('[Error] closeness error, your values do not match the expected values.')
        print('Wrong values for h2')
        print('Your values:    ', h2)
        print('Expected values:', h2_true)
        exit(1)

    print('[Info] GRUCell test: PASS\n')
 
