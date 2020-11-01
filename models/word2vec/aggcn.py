# -*- encoding:utf-8 -*-
'''
@time: 2020/2/12 1:36 下午
@author: huguimin
@email: 718400742@qq.com
'''
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import numpy as np

# max_doc_len = 75
# max_sen_len = 45

class AGClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, embeddings, pos_embedding, opt):
        super().__init__()
        self.gcn_model = AGGCN(embeddings, pos_embedding, opt)
        in_dim = 2 * opt.hidden_dim #200
        self.classifier = nn.Linear(in_dim, opt.num_class) #binary classification
        self.opt = opt

    def forward(self, inputs, attention=False):
        outputs, attention = self.gcn_model(inputs) #(32, 75, 200)
        logits = self.classifier(outputs)
        if not attention:
            return logits
        return logits, attention

class AGGCN(nn.Module):
    def __init__(self, embeddings, pos_embedding, opt):
        super().__init__()
        self.opt = opt
        self.in_dim = opt.embed_dim
        self.emb, self.pos_emb = nn.Embedding.from_pretrained(torch.tensor(embeddings, dtype=torch.float)), \
                                 torch.nn.Embedding(self.opt.pos_num, self.opt.embedding_dim_pos)
        self.mem_dim = 2 * opt.hidden_dim # 200

        # rnn layer
        if self.opt.no_rnn == False:
            self.input_W_R = nn.Linear(self.in_dim, opt.rnn_hidden)
            self.rnn = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=opt.rnn_layer, batch_first=True, bidirectional=True)#(32,75,45,200)
            self.in_dim = opt.hidden_dim * 2
            self.rnn_drop = nn.Dropout(opt.rnn_dropout)  # use on last layer output
        if self.opt.no_pos == False:
            self.in_dim = opt.embed_dim + opt.embedding_dim_pos  # 250
        self.input_W_G = nn.Linear(self.in_dim, self.mem_dim)

        self.in_drop = nn.Dropout(opt.input_dropout)
        self.num_layers = opt.num_layers

        self.clause_encode = Attention(2*opt.hidden_dim, 1, opt.max_sen_len, opt)#(32,75,200)

        self.layers = nn.ModuleList()

        self.heads = opt.nheads
        self.sublayer_first = opt.sublayer_first
        self.sublayer_second = opt.sublayer_second

        # gcn layer
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(GraphConvLayer(opt, self.mem_dim, self.sublayer_first))
                self.layers.append(GraphConvLayer(opt, self.mem_dim, self.sublayer_second))
            else:
                self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_first, self.heads))
                self.layers.append(MultiGraphConvLayer(opt, self.mem_dim, self.sublayer_second, self.heads))

        self.aggregate_W = nn.Linear(len(self.layers) * self.mem_dim, self.mem_dim)

        self.attn = MultiHeadAttention(self.heads, self.mem_dim)

    def pack_sen_len(self, sen_len):
        """
        :param sen_len: [32, 75]
        :return:
        """
        batch_size = sen_len.shape[0]
        up_sen_len = np.zeros([batch_size, self.opt.max_doc_len])
        for i, doc in enumerate(sen_len):
            for j, sen in enumerate(doc):
                if sen == 0:
                    up_sen_len[i][j] = 1
                else:
                    up_sen_len[i][j] = sen

        return torch.tensor(up_sen_len)

    def forward(self, inputs):
        words, sen_len, doc_len, doc_id, emotion_id, adj = inputs
        up_sen_len = self.pack_sen_len(sen_len)
        word_embs = torch.reshape(self.emb(words), [-1, self.opt.max_sen_len, 2*self.opt.hidden_dim]) #(32*75, 45, 200)
        if self.opt.no_rnn == False:
            # word_embs = torch.reshape(self.emb(words), [-1, 2 * self.opt.hidden_dim])  # (32*75, 45, 200)
            # word_embs = self.input_W_R(word_embs) #(32*75, 45, 200)
            up_sen_len = torch.reshape(up_sen_len, [-1])
            word_embs = self.rnn_drop(self.rnn(word_embs, up_sen_len))
        clause_encode = self.clause_encode(word_embs, sen_len)
        embs = [clause_encode]
        if self.opt.no_pos == False:
            embs += [self.pos_emb(torch.LongTensor(position_weight(words, emotion_id, doc_len)).to(self.opt.device))]
            embs = torch.cat(embs, dim=2) #(32, 75, 200), (32, 75, 50)
        embs = self.in_drop(embs)

        gcn_inputs = torch.reshape(embs, [-1, embs.shape[2]])
        gcn_inputs = self.input_W_G(gcn_inputs) #(32, 75, 250)/(32, 75, 200)=>(32*75, 200)
        gcn_inputs = torch.reshape(gcn_inputs, [-1, self.opt.max_doc_len, gcn_inputs.shape[1]]) #(32, 75, 200)

        layer_list = []
        attention_list = []
        outputs = gcn_inputs
        for i in range(len(self.layers)):
            if i < 2:
                outputs = self.layers[i](adj, outputs)
                layer_list.append(outputs)
            else:
                attn_tensor = self.attn(outputs, outputs)
                attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                outputs = self.layers[i](attn_adj_list, outputs)
                layer_list.append(outputs)

        aggregate_out = torch.cat(layer_list, dim=2)
        dcgcn_output = self.aggregate_W(aggregate_out)

        return dcgcn_output, attention_list


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers):
        super(GraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim)) #(300, 150) (450, 150) or (300, 75)
            #(375, 75) ,(450, 75) (525, 75)

        # self.weight_list = self.weight_list.cuda()
        # self.linear_output = self.linear_output.cuda()

    def forward(self, adj, gcn_inputs):
        # adj (32, 75, 75)
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs #(32, 75, 200)
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax) #(,150)
            AxW = AxW + self.weight_list[l](outputs)  # self loop #统一到相同纬度上，元素想加
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2) #()
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2) #（,,300）
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim) #拼接多个head连接在一起用于
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        # self.weight_list = self.weight_list.cuda()
        # self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER) #(batch, token, dim)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def position_weight(inputs, emotion_id, doc_len):
    """
    :param inputs: [32, 75, 200]
    :param emotion_id: [32,]
    :param doc_len: [32]
    :param pos_embedding: [103, 50]
    :return:[32,75,50]
    """
    batch_size, max_len = inputs.shape[0], inputs.shape[1]
    relative_pos = np.zeros((batch_size, max_len))
    for sample in range(batch_size):
        len = doc_len[sample].item()
        for i in range(len):
            relative_pos[sample][i] = i - emotion_id[sample].item() + 69
    return relative_pos



class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn