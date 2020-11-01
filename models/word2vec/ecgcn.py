# -*- encoding:utf-8 -*-
'''
@time: 2019/12/21 8:28 下午
@author: huguimin
@email: 718400742@qq.com
一个doc表示一个样本
'''

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ECGCN(nn.Module):

    def __init__(self, word_embedding, pos_embedding, opt):
        super(ECGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(word_embedding, dtype=torch.float))
        self.pos_embed = nn.Embedding.from_pretrained(torch.tensor(pos_embedding, dtype=torch.float))
        self.word_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)#(32,75,45,200)
        self.clause_encode = Attention(2*opt.hidden_dim, 1, opt.max_sen_len, opt)#(32,75,200)
        # gcn
        # self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        # self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        # self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)

        #gat
        # self.ga1 = GAT(2*opt.hidden_dim, 2*opt.hidden_dim, self.opt.num_class, self.opt.keep_prob1, self.opt.alpha, self.opt.heads)
        self.fc1 = nn.Linear(2*opt.hidden_dim + self.opt.embedding_dim_pos, 2*opt.hidden_dim)
        self.fc2 = nn.Linear(2*opt.hidden_dim, opt.num_class)
        self.text_embed_dropout = nn.Dropout(opt.keep_prob1)

        self.gates = nn.ModuleList()
        self.gcns = nn.ModuleList()
        for i in range(3):
            self.gcns.append(GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim))
            self.gates.append(nn.Linear(2*opt.hidden_dim, 1))


    def position_weight(self, inputs, emotion_id, doc_len):
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

    def emotion_encode(self, inputs, emotion_id):
        """
        :param inputs: [32, 75, 200]
        :param emotion_id: [32,]
        :param doc_len: [32,]
        :return: [32, 1, 200]
        """
        batch_size, max_len, dim = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        emotion_clause = np.zeros((batch_size, dim))

        for sample in range(batch_size):
            clause = inputs[sample][emotion_id[sample]]
            emotion_clause[sample] = clause.cpu().detach().numpy()
        return torch.FloatTensor(emotion_clause)

    def emotion_weight(self, inputs, emotion_clause):
        """
        :param inputs: [32, 75, 200]
               emotion_clause:[32, 1, 200]
        :return: [32, 75]
        """
        batch, dim = inputs.shape[0], inputs.shape[2]
        emotion_clause = torch.reshape(emotion_clause, [batch, dim, 1])
        alpha = torch.reshape(torch.matmul(inputs, emotion_clause.float()), [-1, self.opt.max_doc_len, 1])
        return alpha

    def mask(self, inputs, emotion_id):
        """
        :param inputs: [32,75,200]
        :param emotion_id: [32,]
        :return: [32, 1, 200]
        """
        batch_size, max_len = inputs.shape[0], inputs.shape[1]
        emotion_idx = emotion_id.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(emotion_idx[i]):
                mask[i].append(0)
            for j in range(emotion_idx[i], emotion_id[i] + 1):
                mask[i].append(1)
            for j in range(emotion_idx[i] + 1, max_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * inputs

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
        x, sen_len, doc_len, doc_id, emotion_id, adj = inputs
        up_sen_len = self.pack_sen_len(sen_len)
        x = torch.reshape(x, [-1, self.opt.max_sen_len])
        x = self.embed(x)
        x = self.text_embed_dropout(x)
        up_sen_len = torch.reshape(up_sen_len, [-1])
        word_encode = self.word_lstm(x, up_sen_len) #(32*75, batch_max_len, 200)
        clause_encode = self.clause_encode(word_encode, sen_len)
        embs = [clause_encode]
        embs += [self.pos_embed(torch.LongTensor(self.position_weight(clause_encode, emotion_id, doc_len)).to(self.opt.device))]

        emotion_encode = self.emotion_encode(clause_encode, emotion_id) ###情感子句的嵌入表示
        ###对每层的GCN都与emotion_encode计算一个score.


        # x = F.relu(self.gc1(clause_encode, adj))
        # x = F.relu(self.gc2(x, adj))
        # x = F.relu(self.gc3(x, adj))
        x = clause_encode
        for i in range(3):
            x = F.relu(self.gcns[i](x, adj))
            weight = F.sigmoid(self.gates[i](emotion_encode))
            weight = weight.unsqueeze(dim=-1)
            x = x * weight

        output = self.fc2(x.float())
        return output



    # def forward(self, inputs, vs=False):
    #     attention = []
    #     x, sen_len, doc_len, doc_id, emotion_id, adj = inputs#(x(32,75, 45)), (32, 75)
    #     up_sen_len = self.pack_sen_len(sen_len)
    #     x = torch.reshape(x, [-1, self.opt.max_sen_len])
    #     x = self.embed(x)
    #     x = self.text_embed_dropout(x)
    #     up_sen_len = torch.reshape(up_sen_len, [-1])
    #     word_encode = self.word_lstm(x, up_sen_len) #(32*75, batch_max_len, 200)
    #     clause_encode = self.clause_encode(word_encode, sen_len)
    #     embs = [clause_encode]
    #     embs += [self.pos_embed(torch.LongTensor(self.position_weight(clause_encode, emotion_id, doc_len)).to(self.opt.device))]
    #     "concat"
    #     clause_encode = torch.cat(embs, dim=2)
    #     clause_encode = torch.reshape(clause_encode, [-1, self.opt.max_doc_len, 2 * self.opt.hidden_dim + self.opt.embedding_dim_pos])
    #     clause_encode = self.fc1(clause_encode)
    #     # 策略1 "emotion clause 与 clause的attention weight"
    #     # emotion_encode = self.emotion_encode(clause_encode, emotion_id)
    #     # batch, dim = clause_encode.shape[0], clause_encode.shape[2]
    #     # emotion_encode = torch.reshape(emotion_encode, [batch, dim , 1])
    #     # alpha = self.emotion_weight(clause_encode, emotion_encode)
    #     #
    #     # ones = torch.ones((batch, self.opt.max_doc_len, 1))
    #     #
    #     # emotion_encode = emotion_encode.expand(-1,-1,self.opt.max_doc_len).transpose(1,2)
    #     # clause_encode = alpha * emotion_encode + (ones-alpha)*clause_encode
    #     x = F.relu(self.gc1(clause_encode, adj))
    #     x = F.relu(self.gc2(x, adj))
    #     # x = F.relu(self.gc3(x, adj))
    #     # output = self.ga1(clause_encode, adj)
    #
    #     batch, dim = clause_encode.shape[0], clause_encode.shape[2]
    #     ones = torch.ones((batch, self.opt.max_doc_len, 1)).to(self.opt.device)
    #     emotion_encode = self.emotion_encode(x, emotion_id).to(self.opt.device)
    #     alpha = self.emotion_weight(clause_encode, emotion_encode)
    #     # # emotion_encode = self.mask(x, emotion_id)
    #     # # alpha_mat = torch.matmul(emotion_encode, clause_encode.transpose(1,2))
    #     # # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2).transpose(1,2) #(32,1,75)
    #     # # ones = torch.ones((batch, self.opt.max_doc_len, 1))
    #     # emotion_encode = torch.reshape(emotion_encode, [batch, dim, 1])
    #     # emotion_encode = emotion_encode.expand(-1, -1, self.opt.max_doc_len).transpose(1, 2)
    #     # # x = emotion_encode * alpha + (ones-alpha)*clause_encode
    #     emotion_encode = torch.reshape(emotion_encode, [batch, dim, 1])
    #     emotion_encode = emotion_encode.expand(-1, -1, self.opt.max_doc_len).transpose(1, 2)
    #     x = clause_encode * alpha + (ones - alpha) * emotion_encode
    #     x = self.text_embed_dropout(x)
    #     # # x = torch.matmul(alpha, clause_encode).squeeze(1)
    #     #
    #     # # 策略2 以原始的句表示为主，图卷积作为辅助
    #     # #
    #     #
    #     output = self.fc2(x.float())
    #     if vs:
    #         return output, attention
    #     return output












