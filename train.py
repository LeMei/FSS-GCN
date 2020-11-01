# -*- encoding:utf-8 -*-
'''
@time: 2019/12/21 9:48 下午
@author: huguimin
@email: 718400742@qq.com
'''
import os
import random
import math
import torch
import argparse
import numpy as np
from util.util_data_gcn import *
from models.word2vec.ecgcn import ECGCN
from models.word2vec.ecgat import ECGAT
from models.word2vec.fssgcn import ECClassifier
from models.word2vec.aggcn import AGClassifier
# from models.ecaggcn_no_dcn import ECClassifier
from sklearn import metrics
import torch.nn as nn
import time

class Model:

    def __init__(self, opt, idx):
        self.opt = opt

        self.embedding = load_embedding(opt.embedding_path)
        self.embedding_pos = load_pos_embedding(opt.embedding_dim_pos)
        self.split_size = math.ceil(opt.data_size / opt.n_split)

        self.global_f1 = 0
        # self.train, self.test = load_data(self.split_size, idx, opt.data_size) #意味着只能从一个角度上训练，应该换几种姿势轮着训练
        if opt.dataset == 'EC':
            self.train, self.test = load_percent_train(opt.per, self.split_size, idx, opt.data_size)
        elif opt.dataset == 'EC_en':
            self.train, self.test = load_data_en()
        else:
            print('DATASET NOT EXIST')
        # self.train, self.test = load_data(self.split_size, idx, opt.data_size)
        self.sub_model = opt.model_class(self.embedding, self.embedding_pos, self.opt).to(opt.device)


    def _reset_params(self):
        for p in self.sub_model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _print_args(self):
        n_trainable_params, n_nontrainable_params, model_params = 0, 0, 0
        for p in self.sub_model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            model_params += n_params
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}, model_params: {2}'.format(n_trainable_params, n_nontrainable_params, model_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _train(self, criterion, optimizer):

        max_test_pre = 0
        max_test_rec = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False

            for train in get_train_batch_data(self.train, self.opt.batch_size, self.opt.keep_prob1, self.opt.keep_prob2):
                global_step += 1
                self.sub_model.train()
                optimizer.zero_grad()

                inputs = [train[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = train['label'].to(self.opt.device)
                doc_len = train['doc_len'].to(self.opt.device)
                targets = torch.argmax(targets, dim=2)
                targets_flatten = torch.reshape(targets, [-1])

                outputs = self.sub_model(inputs)
                outputs_flatten = torch.reshape(outputs, [-1, self.opt.num_class])
                loss = criterion(outputs_flatten, targets_flatten)
                # loss = nn.functional.nll_loss(outputs_flatten, targets_flatten)
                outputs = torch.argmax(outputs, dim=-1)

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    train_acc, train_pre, train_rec, train_f1 = self._evaluate_prf_binary(targets, outputs, doc_len)
                    print('Train: loss:{:.4f}, train_acc: {:.4f}, train_pre:{:.4f}, train_rec:{:.4f}, train_f1: {:.4f}\n'.format(loss.item(), train_acc, train_pre, train_rec, train_f1))

                    test_acc, test_pre, test_rec, test_f1 = self._evaluate_acc_f1()
                    # if test_acc > max_test_acc:
                    #     max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        max_test_pre = test_pre
                        max_test_rec = test_rec
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(self.sub_model.state_dict(), 'state_dict/'+self.opt.model_name+'_'+self.opt.dataset+'_test.pkl')
                            print('>>> best model saved.')
                    print('Test: test_acc: {:.4f}, test_pre:{:.4f}, test_rec:{:.4f}, test_f1: {:.4f}'.format(test_acc, test_pre, test_rec, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0
        return max_test_pre, max_test_rec, max_test_f1

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.sub_model.eval()
        targets_all, outputs_all, doc_len_all = None, None, None
        inference_time_list = []
        with torch.no_grad():
            for test in get_test_batch_data(self.test, self.opt.batch_size):
                inputs = [test[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = test['label'].to(self.opt.device)
                doc_len = test['doc_len'].to(self.opt.device)
                targets = torch.argmax(targets, dim=2)#(32,75)
                if self.opt.infer_time:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    outputs = self.sub_model(inputs)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    inference_time = end_time - start_time
                    inference_time_list.append(inference_time/targets.shape[0])
                else:
                    outputs = self.sub_model(inputs)
                outputs = torch.argmax(outputs, dim=2)#(32, 75)

                if targets_all is None:
                    targets_all = targets
                    outputs_all = outputs
                    doc_len_all = doc_len
                else:
                    targets_all = torch.cat((targets_all, targets), dim=0)
                    outputs_all = torch.cat((outputs_all, outputs), dim=0)
                    doc_len_all = torch.cat((doc_len_all, doc_len), dim=0)



            test_acc, test_pre, test_rec, test_f1 = self._evaluate_prf_binary(targets_all, outputs_all, doc_len_all)
        infer_time = np.mean(np.array(inference_time_list))
        print('infer_time==================', str(infer_time))
        return test_acc, test_pre, test_rec, test_f1

    def _evaluate_prf_binary(self, targets, outputs, doc_len):
        """
        :param targets: [32,75]
        :param outputs: [32,75]
        :return:
        """
        tmp1, tmp2 = [], []
        for i in range(outputs.shape[0]):
            for j in range(doc_len[i]):
                tmp1.append(outputs[i][j].cpu())
                tmp2.append(targets[i][j].cpu())
        y_pred, y_true = np.array(tmp1), np.array(tmp2)
        acc = metrics.precision_score(y_true, y_pred, average='micro')
        p = metrics.precision_score(y_true, y_pred, average='binary')
        r = metrics.recall_score(y_true, y_pred, average='binary')
        f1 = metrics.f1_score(y_true, y_pred, average='binary')
        return acc, p, r, f1


    def run(self, folder, repeats=1):
        # Loss and Optimizer
        print(('-'*50 + 'Folder{}' + '-'*50).format(folder))
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.functional.nll_loss()
        _params = filter(lambda p: p.requires_grad, self.sub_model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/' + self.opt.model_name + '_' + str(folder) + '_test.txt', 'a+', encoding='utf-8')

        max_test_pre_avg = 0
        max_test_rec_avg = 0
        max_test_f1_avg = 0
        for i in range(repeats):
            print('repeat: ', (i + 1))
            f_out.write('repeat: ' + str(i + 1))
            self._reset_params()
            max_test_pre, max_test_rec, max_test_f1 = self._train(criterion, optimizer)
            print('max_test_acc: {0}     max_test_hf1: {1}'.format(max_test_pre, max_test_f1))
            f_out.write('max_test_acc: {0}, max_test_f1: {1}'.format(max_test_pre, max_test_f1))
            max_test_pre_avg += max_test_pre
            max_test_rec_avg += max_test_rec
            max_test_f1_avg += max_test_f1
            print('#' * 100)
        print("max_test_acc_avg: {.4f}", max_test_pre_avg / repeats)
        print('max_test_acc_rec: {.4f}', max_test_rec_avg / repeats)
        print("max_test_f1_avg: {.4f}", max_test_f1_avg / repeats)
        f_out.write("max_test_pre_avg: {0}, max_test_rec_avg: {1}, max_test_f1_avg: {2}".format(max_test_pre_avg / repeats, max_test_rec_avg / repeats, max_test_f1_avg / repeats))
        f_out.close()
        return max_test_pre_avg / repeats, max_test_rec_avg / repeats, max_test_f1_avg / repeats

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='fssgcn', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--input_dropout', default=0.1, type=float)
    parser.add_argument('--gcn_dropout', default=0.1, type=float)
    parser.add_argument('--head_dropout', default=0.1, type=float)
    parser.add_argument('--keep_prob2', default=0.1, type=float)
    parser.add_argument('--keep_prob1', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.3, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    # parser.add_argument('--l2reg', default=0.000005, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=200, type=int)
    parser.add_argument('--embedding_dim_pos', default=100, type=int)
    ###中文数据集的embedding文件
    parser.add_argument('--embedding_path', default='embedding.txt', type=str)
    ###英文数据集的embedding文件################################
    # parser.add_argument('--embedding_path', default='all_embedding_en.txt', type=str)
    #################################################

    parser.add_argument('--pos_num',default=138, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)

    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--nheads', default=1, type=int)
    parser.add_argument('--sublayer_first', default=2, type=int)
    parser.add_argument('--sublayer_second', default=4, type=int)
    parser.add_argument('--sublayer', default=1, type=int)

    parser.add_argument('--no_rnn', default=False, type=bool)
    parser.add_argument('--rnn_layer', default=1, type=int)
    parser.add_argument('--rnn_hidden', default=100, type=int)
    parser.add_argument('--rnn_dropout', default=0.5, type=float)

    parser.add_argument('--no_pos', default=False, type=bool)
    parser.add_argument('--n_split', default=10, type=int)
    parser.add_argument('--per', default=1.0, type=float)

    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--infer_time', default=False, type=bool)
    ####数据集为英文数据集
    # parser.add_argument('--dataset', default='EC_en', type=str)

    ####数据集为中文数据集
    parser.add_argument('--dataset', default='EC', type=str)

    opt = parser.parse_args()

    model_classes = {
        'ecgcn': ECGCN,
        'ecgat': ECGAT,
        'aggcn': AGClassifier,
        'fssgcn': ECClassifier
    }
    input_colses = {
        'ecgcn': ['content', 'sen_len', 'doc_len', 'doc_id', 'emotion_id', 'graph'],
        'ecgat': ['content', 'sen_len', 'doc_len', 'doc_id', 'emotion_id', 'graph'],
        'aggcn': ['content', 'sen_len', 'doc_len', 'doc_id', 'emotion_id', 'graph'],
        'fssgcn': ['content', 'sen_len', 'doc_len', 'doc_id', 'emotion_id', 'graph']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    if opt.dataset == 'EC':
        opt.max_doc_len = 75
        opt.max_sen_len = 45
        opt.data_size = 2105
        opt.hidden_dim = 100
        opt.rnn_hidden = 100
        opt.embed_dim = 200
        opt.embedding_path = 'embedding.txt'
    else:
        opt.max_doc_len = 45
        opt.max_sen_len = 130
        opt.data_size = 2105
        opt.hidden_dim = 150
        opt.rnn_hidden = 150
        opt.embed_dim = 300
        opt.embedding_path = 'all_embedding_en.txt'

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    p, r, f1 = [], [], []

    for i in range(1):
        model = Model(opt, i)
        ###计算模型大
        model._print_args()
        p_t, r_t, f1_t = model.run(i)
        p.append(p_t)
        r.append(r_t)
        f1.append(f1_t)
    print("max_test_pre_avg: {:.4f}, max_test_rec_avg: {:.4f}, max_test_f1_avg: {:.4f}".format(np.mean(p), np.mean(r), np.mean(f1)))











