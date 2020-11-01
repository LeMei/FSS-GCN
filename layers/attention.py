# -*- encoding:utf-8 -*-
'''
@time: 2019/12/21 8:58 下午
@author: huguimin
@email: 718400742@qq.com
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# max_sen_len = 45
# max_doc_len = 75

class Attention(nn.Module):
    def __init__(self, feature_in, feature_out, sen_len, opt):
        super(Attention, self).__init__()
        self.feature_in = feature_in#200
        self.feature_out = feature_out#200
        self.sen_len = sen_len #(32, 75)
        self.weight_1 = nn.Parameter(torch.FloatTensor(feature_in, feature_in))
        self.bias_1 = nn.Parameter(torch.FloatTensor(feature_in))
        self.weight_2 = nn.Parameter(torch.FloatTensor(feature_in, feature_out))

        self.opt = opt


    def sequence_mask(self, lengths, maxlen, out_shape, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.opt.device)
        matrix = torch.unsqueeze(lengths, dim=-1).to(self.opt.device)
        mask = row_vector < matrix

        mask.type(dtype)
        return torch.reshape(mask, out_shape)

    def forward(self, inputs, sen_len):
        """
        :param inputs: (batch_size*max_doc_len, max_sen_len, dim)
        :param max_sen_len:
        :return: (batch_size, max_doc_len, dim)
        """
        max_len, dim = inputs.shape[1], inputs.shape[2]
        inputs_tmp = torch.reshape(inputs, [-1, dim])
        u = torch.tanh(torch.matmul(inputs_tmp, self.weight_1) + self.bias_1) #(batch_size*max_doc_len*max_sen_len, dim)
        alpha = torch.reshape(torch.matmul(u, self.weight_2), [-1, 1, max_len])#(batch_size*max_doc_len,  1, max_len)
        # alpha = torch.exp(alpha)
        alpha = F.softmax(alpha, dim=2)
        alpha = alpha * self.sequence_mask(sen_len, max_len, alpha.shape)
        return torch.matmul(alpha, inputs).reshape([-1, self.opt.max_doc_len, dim])

