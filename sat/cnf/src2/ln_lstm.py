from torch.nn import Module, Parameter, LayerNorm, Dropout
import torch


def decode_fn(name):
    if name == 'tanh':
        return torch.tanh
    elif name == 'sigmoid':
        return torch.sigmoid
    elif name == 'relu':
        return torch.relu
    elif name == 'tan':
        return torch.tan
    return None


class ln_LSTMCell(Module):
    '''
    layer-nrom lstm
    port of https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/rnn/python/ops/rnn_cell.py

    '''

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 forget_bias=1.0, dropout: float = 0.,
                 layer_norm: bool = True,  # norm_gain: float = 1.0, norm_shift: float = 0.0,
                 activation: str = 'tanh'):
        super(ln_LSTMCell, self).__init__()

        self._num_units = input_size
        self._num_hidden_units = hidden_size
        self._activation = decode_fn(activation)
        self._forget_bias = forget_bias
        self._dropout = dropout  # 1-_keep_prob

        iu = hidden_size ** -0.5
        # the learnable input-hidden weights layer (W_ii|W_if|W_ig|W_io;W_hi|W_hf|W_hg|W_ho),
        # of shape (4*hidden_size, input_size) for k = 0. Otherwise,
        # the shape is (hidden_size, 4*hidden_size)
        self.weight_ihh = Parameter(torch.empty((self._num_units + self._num_hidden_units, 4 * self._num_hidden_units,),
                                                dtype=torch.float32), True)
        torch.nn.init.uniform_(self.weight_ihh, -iu, iu)
        self._bias = bias
        if bias:
            # the learnable input-hidden and hidden-hidden bias
            # (b_ihi|b_ihf|b_ihg|b_iho), of shape (4*hidden_size)
            self.bias_ihh = Parameter(torch.empty((4 * self._num_hidden_units,), dtype=torch.float32), True)

            torch.nn.init.uniform_(self.bias_ihh, -iu, iu)

        self._layer_norm = layer_norm
        # self._norm_gain = norm_gain
        # self._norm_shift = norm_shift

        if self._layer_norm:
            self.i_norm = LayerNorm([self._num_hidden_units])
            self.f_norm = LayerNorm([self._num_hidden_units])
            self.g_norm = LayerNorm([self._num_hidden_units])
            self.o_norm = LayerNorm([self._num_hidden_units])

        self.g_dropout = Dropout(self._dropout)

    def forward(self, input, hx=None):

        batch_len = input.shape[0]
        elem_len = input.shape[1]

        if isinstance(hx, tuple):
            hid_in, cstate_in = hx
        else:
            hid_in, cstate_in = None, None
        if hid_in is None:
            hid_in = torch.zeros_like(input)

        if cstate_in is None:
            cstate_in = torch.zeros_like(input)

        sig = torch.sigmoid
        act = self._activation

        ifgo_lin = torch.matmul(torch.cat([input, hid_in], 1), self.weight_ihh)
        if self._bias:
            ifgo_lin += self.bias_ihh

        i = ifgo_lin[:, :self._num_hidden_units]
        f = ifgo_lin[:, self._num_hidden_units:2 * self._num_hidden_units]
        g = ifgo_lin[:, 2 * self._num_hidden_units:3 * self._num_hidden_units]
        o = ifgo_lin[:, 3 * self._num_hidden_units:4 * self._num_hidden_units]

        if self._layer_norm:
            i = self.i_norm(i)
            f = self.f_norm(f)
            g = self.g_norm(g)
            o = self.o_norm(o)

        i = sig(i)
        f = sig(f)
        g = act(g)
        o = sig(o)

        if self._dropout > 0:
            g = self.g_dropout(g)

        c_state_out = f * cstate_in + i * g
        hid_out = o * act(c_state_out)

        return hid_out, c_state_out


class ln_LSTMCell2(Module):
    '''
    layer-nrom lstm
    port of https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/contrib/rnn/python/ops/rnn_cell.py

    '''

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 forget_bias=1.0, dropout: float = 0.,
                 layer_norm: bool = True,  # norm_gain: float = 1.0, norm_shift: float = 0.0,
                 activation: str = 'tanh'):
        super(ln_LSTMCell2, self).__init__()

        self._num_units = input_size
        self._num_hidden_units = hidden_size
        self._activation = decode_fn(activation)
        self._forget_bias = forget_bias
        self._dropout = dropout  # 1-_keep_prob

        iu = hidden_size ** -0.5
        # the learnable input-hidden weights layer (W_ii|W_if|W_ig|W_io;W_hi|W_hf|W_hg|W_ho),
        # of shape (4*hidden_size, input_size) for k = 0. Otherwise,
        # the shape is (hidden_size, 4*hidden_size)
        self.weight_ihh = Parameter(torch.empty((self._num_units + self._num_hidden_units, 4 * self._num_hidden_units,),
                                                dtype=torch.float32), True)
        torch.nn.init.uniform_(self.weight_ihh, -iu, iu)
        self._bias = bias
        if bias:
            # the learnable input-hidden and hidden-hidden bias
            # (b_ihi|b_ihf|b_ihg|b_iho), of shape (4*hidden_size)
            # self.i_bias = Parameter(torch.empty((self._num_hidden_units,), dtype=torch.float32), True)
            # torch.nn.init.uniform_(self.i_bias, -iu, iu)
            self.f_bias = Parameter(torch.empty((self._num_hidden_units,), dtype=torch.float32), True)
            torch.nn.init.uniform_(self.f_bias, -iu, iu)
            # self.g_bias = Parameter(torch.empty((self._num_hidden_units,), dtype=torch.float32), True)
            # torch.nn.init.uniform_(self.g_bias, -iu, iu)
            # self.o_bias = Parameter(torch.empty((self._num_hidden_units,), dtype=torch.float32), True)
            # torch.nn.init.uniform_(self.o_bias, -iu, iu)

        self._layer_norm = layer_norm
        # self._norm_gain = norm_gain
        # self._norm_shift = norm_shift

        if self._layer_norm:
            self.i_norm = LayerNorm([self._num_hidden_units])
            self.f_norm = LayerNorm([self._num_hidden_units])
            self.g_norm = LayerNorm([self._num_hidden_units])
            self.o_norm = LayerNorm([self._num_hidden_units])
            self.c_norm = LayerNorm([self._num_hidden_units])

        self.g_dropout = Dropout(self._dropout)

    def forward(self, input, hx=None):


        if isinstance(hx, tuple):
            hid_in, cstate_in = hx
        else:
            hid_in, cstate_in = None, None
        if hid_in is None:
            hid_in = torch.zeros_like(input)

        if cstate_in is None:
            cstate_in = torch.zeros_like(input)

        sig = torch.sigmoid
        act = self._activation

        ifgo_lin = torch.matmul(torch.cat([input, hid_in], 1), self.weight_ihh)

        i = ifgo_lin[:, :self._num_hidden_units]
        f = ifgo_lin[:, self._num_hidden_units:2 * self._num_hidden_units]
        g = ifgo_lin[:, 2 * self._num_hidden_units:3 * self._num_hidden_units]
        o = ifgo_lin[:, 3 * self._num_hidden_units:4 * self._num_hidden_units]

        if self._layer_norm:
            i = self.i_norm(i)
            f = self.f_norm(f)
            g = self.g_norm(g)
            o = self.o_norm(o)

        # if self._bias:
        #     i += self.i_bias
        #     f += self.f_bias
        #     g += self.g_bias
        #     o += self.o_bias

        g = act(g)
        if self._dropout > 0:
            g = self.g_dropout(g)

        c_state_out = sig(f + self.f_bias) * cstate_in + sig(i) * g

        if self._layer_norm:
            c_state_out = self.c_norm(c_state_out)

        hid_out = sig(o) * act(c_state_out)

        return hid_out, c_state_out
