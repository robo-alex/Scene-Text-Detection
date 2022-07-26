#========================================================
#             RNNCell.py - implementation of Gated Recurrent Unit
#========================================================

import math
import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.activation = nn.Tanh()

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

        new_hidden = self.activation(torch.matmul(input, self.weight_ih) + self.bias_ih + torch.matmul(hidden, self.weight_hh) + self.bias_hh)# new_hidden = ???
        return new_hidden


if __name__ == '__main__':
    # define the RNN model
    rnn = RNNCell(input_size=2, hidden_size=3)

    with torch.no_grad():
        # set model's parameter
        rnn.weight_ih.copy_(torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float))
        rnn.bias_ih.copy_(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float))
        rnn.weight_hh.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float))
        rnn.bias_hh.copy_(torch.tensor([-0.1, -0.2, 0.3], dtype=torch.float))

    h0 = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float)
    x1 = torch.tensor([[0.1, -0.2], [-0.7, -0.2]], dtype=torch.float)
    x2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float)

    # process for model forwarding 
    h1 = rnn(x1, h0)
    h2 = rnn(x2, h1)
    
    print('[Info] testing RNNCell...')
    h1_true = torch.load('data/RNNCell_results/h1.pth')
    h2_true = torch.load('data/RNNCell_results/h2.pth')
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

    print('[Info] RNNCell test: PASS\n')

    
