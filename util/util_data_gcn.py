# -*- encoding:utf-8 -*-
'''
@time: 2019/12/21 5:11 下午
@author: huguimin
@email: 718400742@qq.com
'''
import pickle as pk
import numpy as np
import random
import torch
path = './data/'
max_doc_len = 75
max_sen_len = 45
max_word_len = 430
batch_size = 32

###数据集
###train
# max_doc_len = 45
# max_sen_len = 55

###test

# max_sen_len = 130
# max_doc_len = 40
###主要是将english 文本处理为和中文文本相同的格式的


def pad_graph(file_name):
    # fin = open(file_name + '_phrase.graph', 'rb')
    fin = open(file_name, 'rb')
    idx2gragh = pk.load(fin)
    fin.close()
    graph = []
    for i, doc in enumerate(idx2gragh):
        adj_matrix = idx2gragh[i+1]
        doc_len = len(adj_matrix[0])
        adj_matrix = np.pad(adj_matrix, ((0,max_doc_len-doc_len),(0,max_doc_len-doc_len)))
        adj_matrix = np.where(adj_matrix, 1.0, 0.0)
        graph.append(adj_matrix)

    return np.array(graph)

def pad_graph_en(file_name):
    # fin = open(file_name + '_phrase.graph', 'rb')

    ##test


    max_doc_len = 45

    fin = open(file_name, 'rb')
    idx2gragh = pk.load(fin)
    fin.close()
    graph = []
    for i, doc in enumerate(idx2gragh):
        adj_matrix = idx2gragh[i]
        doc_len = len(adj_matrix[0])
        adj_matrix = np.pad(adj_matrix, ((0,max_doc_len-doc_len),(0,max_doc_len-doc_len)))
        adj_matrix = np.where(adj_matrix, 1.0, 0.0)
        graph.append(adj_matrix)

    return np.array(graph)

def set_self(adj_matrix, doc_len):
    for i in range(doc_len):
        max = np.max(adj_matrix[i,:])
        adj_matrix[i][i] = max

def load_embedding(embedding_path):
    word_embedding = pk.load(open(path + embedding_path,'rb'))
    return word_embedding

def load_pos_embedding(embedding_dim_pos):
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(
        loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(-100, 100)])
    return np.array(embedding_pos)

def load_doc(id):
    x = pk.load(open(path + 'x.txt', 'rb'))
    y = pk.load(open(path + 'y.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    doc_id = pk.load(open(path + 'doc_id.txt', 'rb'))
    emotion_id = pk.load(open(path + 'emotion_id.txt', 'rb'))
    graph = pad_graph(path + 'data.csv.graph')
    idx = list(doc_id).index(id)
    data = {
        'content': torch.tensor(x[idx][np.newaxis,:,:]).long(),
        'label': torch.tensor(y[idx][np.newaxis,:,:]),
        'sen_len': torch.tensor(sen_len[idx][np.newaxis,:]),
        'doc_len': torch.tensor([doc_len[idx]]),
        'doc_id': torch.tensor([doc_id[idx]]),
        'emotion_id': torch.tensor([emotion_id[idx]]),
        'graph': torch.tensor([graph[idx]]).float(),
        'keep_prob1': 1.0,
        'keep_prob2': 1.0
    }
    return data
def load_inter_data(start, end):
    """
    0-10, 332
    10-20, 1452
    20-30, 280
    30-40, 35
    40-50, 4
    50-2

    0-5, 52
    5-10, 415
    10-15, 849
    15-20, 518
    20-25,181
    25-30, 56
    30-35,25
    35- 9
    """
    x = pk.load(open(path + 'x.txt', 'rb'))
    y = pk.load(open(path + 'y.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    doc_id = pk.load(open(path + 'doc_id.txt', 'rb'))
    emotion_id = pk.load(open(path + 'emotion_id.txt', 'rb'))
    graph = pad_graph(path + 'data.csv.graph')
    index = np.where((doc_len > start) & (doc_len <= end))
    # test = {
    #     'content': x[index],
    #     'label': y[index],
    #     'sen_len': sen_len[index],
    #     'doc_len': doc_len[index],
    #     'doc_id': doc_id[index],
    #     'emotion_id': emotion_id[index],
    #     'graph': graph[index]
    # }
    return doc_id[index]

def load_percent_train(per, split_size, start_index, data_size):
    x = pk.load(open(path + 'x.txt', 'rb'))
    y = pk.load(open(path + 'y.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    doc_id = pk.load(open(path + 'doc_id.txt', 'rb'))
    emotion_id = pk.load(open(path + 'emotion_id.txt', 'rb'))
    graph = pad_graph(path + 'data.csv.graph')

    length = int(per * data_size)
    mid = start_index * split_size
    right = min((start_index + 1) * split_size, data_size)
    left = max((start_index - 1) * split_size, 0)
    x_test, y_test = map(lambda d: d[mid:right, :, :], [x, y])
    sen_len_test = sen_len[mid:right, :]
    doc_len_test, doc_id_test, emotion_id_test = \
        map(lambda d: d[mid:right], [doc_len, doc_id, emotion_id])
    graph_test = graph[mid:right]
    graph_train = np.vstack((graph[0:mid], graph[right:data_size]))[0:length]
    x_train, y_train = map(lambda d: d[:length], map(lambda d: np.vstack((d[0:mid, :, :], d[right:data_size, :, :])), [x, y]))
    sen_len_train = np.vstack((sen_len[0:mid, :], sen_len[right:data_size, :]))[:length]
    doc_len_train, doc_id_train, emotion_id_train = map(lambda d: d[:length], map(lambda d: np.hstack((d[0:mid], d[right:data_size])), [doc_len, doc_id, emotion_id]))
    train = {
        'content': x_train,
        'label': y_train,
        'sen_len': sen_len_train,
        'doc_len': doc_len_train,
        'doc_id': doc_id_train,
        'emotion_id': emotion_id_train,
        'graph': graph_train
    }
    test = {
        'content': x_test,
        'label': y_test,
        'sen_len': sen_len_test,
        'doc_len': doc_len_test,
        'doc_id': doc_id_test,
        'emotion_id': emotion_id_test,
        'graph': graph_test
    }
    return train, test

def load_data_en():
    x_train = pk.load(open(path + 'train_x_en.txt', 'rb'))
    y_train = pk.load(open(path + 'train_y_en.txt', 'rb'))
    sen_len_train = pk.load(open(path + 'train_sen_len_en.txt', 'rb'))
    doc_len_train = pk.load(open(path + 'train_doc_len_en.txt', 'rb'))
    doc_id_train = pk.load(open(path + 'train_doc_id_en.txt', 'rb'))
    emotion_id_train = pk.load(open(path + 'train_emotion_id_en.txt', 'rb'))
    graph_train = pad_graph_en(path + 'ntcir_eng_train_pre.txt_en.graph')
    print(
        'x.shape {} \ny.shape {}\nsen_len.shape {} \ndoc_len.shape {}\ndoc_id.shape {}\nemotion_id.shape{}\ngraph.shape{}'
            .format(x_train.shape, y_train.shape, sen_len_train.shape, doc_len_train.shape, doc_id_train.shape, emotion_id_train.shape, graph_train.shape))

    train = {
        'content': x_train,
        'label': y_train,
        'sen_len': sen_len_train,
        'doc_len': doc_len_train,
        'doc_id': doc_id_train,
        'emotion_id': emotion_id_train,
        'graph': graph_train
    }

    x_test = pk.load(open(path + 'test_x_en.txt', 'rb'))
    y_test = pk.load(open(path + 'test_y_en.txt', 'rb'))
    sen_len_test = pk.load(open(path + 'test_sen_len_en.txt', 'rb'))
    doc_len_test = pk.load(open(path + 'test_doc_len_en.txt', 'rb'))
    doc_id_test = pk.load(open(path + 'test_doc_id_en.txt', 'rb'))
    emotion_id_test = pk.load(open(path + 'test_emotion_id_en.txt', 'rb'))
    graph_test = pad_graph_en(path + 'ntcir_eng_test_pre.txt_en.graph')
    print(
        'x.shape {} \ny.shape {}\nsen_len.shape {} \ndoc_len.shape {}\ndoc_id.shape {}\nemotion_id.shape{}\ngraph.shape{}'
            .format(x_test.shape, y_test.shape, sen_len_test.shape, doc_len_test.shape, doc_id_test.shape,
                    emotion_id_test.shape, graph_test.shape))

    test = {
        'content': x_test,
        'label': y_test,
        'sen_len': sen_len_test,
        'doc_len': doc_len_test,
        'doc_id': doc_id_test,
        'emotion_id': emotion_id_test,
        'graph': graph_test
    }
    return train, test

def load_data(split_size, start_index, data_size):
    x = pk.load(open(path + 'x.txt', 'rb'))
    y = pk.load(open(path + 'y.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    doc_id = pk.load(open(path + 'doc_id.txt', 'rb'))
    emotion_id = pk.load(open(path + 'emotion_id.txt', 'rb'))
    print(
        'x.shape {} \ny.shape {}\nsen_len.shape {} \ndoc_len.shape {}\ndoc_id.shape {}\nemotion_id.shape{}'
        .format(x.shape, y.shape, sen_len.shape, doc_len.shape, doc_id.shape, emotion_id.shape))

    # graph_train = pad_graph(path + 'train_data.txt')
    # graph_test = pad_graph(path + 'test_data.txt')
    # graph = pad_graph(path + 'data.csv')
    graph = pad_graph(path + 'data.csv.graph')
    # x_train, y_train = map(lambda d: d[:1894,:,:], [x, y])
    # sen_len_train = sen_len[:1894,:]
    # doc_len_train, doc_id_train, emotion_id_train = \
    #     map(lambda d: d[:1894], [doc_len, doc_id, emotion_id])
    # graph_train = graph[:1894]
    # graph_test = graph[1894:2105]
    # x_test, y_test =  map(lambda d: d[1894:2105,:,:], [x, y])
    # sen_len_test = sen_len[1894:2105,:]
    # doc_len_test, doc_id_test, emotion_id_test = \
    #     map(lambda d: d[1894:2105], [doc_len, doc_id, emotion_id])
    # x_test, y_test = map(lambda d: d[:211,:,:], [x, y])
    # sen_len_test = sen_len[:211,:]
    # doc_len_test, doc_id_test, emotion_id_test = \
    #     map(lambda d: d[:211], [doc_len, doc_id, emotion_id])
    # graph_test = graph[:211]
    # graph_train = graph[211:2105]
    # x_train, y_train =  map(lambda d: d[211:2105,:,:], [x, y])
    # sen_len_train = sen_len[211:2105,:]
    # doc_len_train, doc_id_train, emotion_id_train = \
    #     map(lambda d: d[211:2105], [doc_len, doc_id, emotion_id])
    mid = start_index * split_size
    right = min((start_index + 1) * split_size, data_size)
    left = max((start_index - 1) * split_size, 0)
    x_test, y_test = map(lambda d: d[mid:right, :, :], [x, y])
    sen_len_test = sen_len[mid:right, :]
    doc_len_test, doc_id_test, emotion_id_test = \
        map(lambda d: d[mid:right], [doc_len, doc_id, emotion_id])
    graph_test = graph[mid:right]
    graph_train = np.vstack((graph[0:mid], graph[right:data_size]))
    x_train, y_train = map(lambda d: np.vstack((d[0:mid,:,:], d[right:data_size,:,:])), [x, y])
    sen_len_train = np.vstack((sen_len[0:mid, :], sen_len[right:data_size, :]))
    doc_len_train, doc_id_train, emotion_id_train = \
        map(lambda d: np.hstack((d[0:mid], d[right:data_size])), [doc_len, doc_id, emotion_id])

    train = {
        'content': x_train,
        'label': y_train,
        'sen_len':sen_len_train,
        'doc_len':doc_len_train,
        'doc_id':doc_id_train,
        'emotion_id':emotion_id_train,
        'graph': graph_train
    }
    test = {
        'content': x_test,
        'label': y_test,
        'sen_len': sen_len_test,
        'doc_len': doc_len_test,
        'doc_id': doc_id_test,
        'emotion_id': emotion_id_test,
        'graph': graph_test
    }
    return train, test

def load_all_data():
    x = pk.load(open(path + 'x.txt', 'rb'))
    y = pk.load(open(path + 'y.txt', 'rb'))
    sen_len = pk.load(open(path + 'sen_len.txt', 'rb'))
    doc_len = pk.load(open(path + 'doc_len.txt', 'rb'))
    doc_id = pk.load(open(path + 'doc_id.txt', 'rb'))
    emotion_id = pk.load(open(path + 'emotion_id.txt', 'rb'))
    print(
        'x.shape {} \ny.shape {}\nsen_len.shape {} \ndoc_len.shape {}\ndoc_id.shape {}\nemotion_id.shape{}'
        .format(x.shape, y.shape, sen_len.shape, doc_len.shape, doc_id.shape, emotion_id.shape))
    graph = pad_graph(path + 'data.csv.graph')
    data = {
        'content': x,
        'label': y,
        'sen_len':sen_len,
        'doc_len':doc_len,
        'doc_id':doc_id,
        'emotion_id':emotion_id,
        'graph': graph
    }
    return data

def batch_index(length, batch_size, test=False):
    index = list(range(length))
    if not test:
        random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size: (i + 1) * batch_size]
        if not test and len(ret) < batch_size:
            break
        yield ret


def get_train_batch_data(train_data, batch_size, keep_prob1, keep_prob2):

    x_train, y_train, sen_len_train, doc_len_train, doc_id_train, emotion_id_train, graph_train = \
        train_data['content'],train_data['label'],train_data['sen_len'],train_data['doc_len'],train_data['doc_id'],train_data['emotion_id'],train_data['graph']
    for index in batch_index(len(y_train), batch_size):
        feed_list = {
            'content': torch.tensor(x_train[index]).long(),
            'label': torch.tensor(y_train[index]),
            'sen_len': torch.tensor(sen_len_train[index]),
            'doc_len': torch.tensor(doc_len_train[index]),
            'doc_id': torch.tensor(doc_id_train[index]),
            'emotion_id': torch.tensor(emotion_id_train[index]),
            'graph': torch.tensor(graph_train[index]).float(),
            'keep_prob1':keep_prob1,
            'keep_prob2':keep_prob2
        }
        yield feed_list

def get_test_batch_data(test_data, batch_size):
    x_test, y_test, sen_len_test, doc_len_test, doc_id_test, emotion_id_test, graph_test = \
        test_data['content'],test_data['label'],test_data['sen_len'],test_data['doc_len'],test_data['doc_id'],test_data['emotion_id'],test_data['graph']
    for index in batch_index(len(y_test), batch_size, test=True):
        feed_list = {
            'content': torch.tensor(x_test[index]).long(),
            'label': torch.tensor(y_test[index]),
            'sen_len': torch.tensor(sen_len_test[index]),
            'doc_len': torch.tensor(doc_len_test[index]),
            'doc_id': torch.tensor(doc_id_test[index]),
            'emotion_id': torch.tensor(emotion_id_test[index]),
            'graph': torch.tensor(graph_test[index]).float(),
            'keep_prob1': 1.0,
            'keep_prob2': 1.0
        }
        yield feed_list

def pack_data(x, doc_len):
    """
    :param x: (batch_size, max_doc_len, max_sen_len)
    :param sen_len:
    :param doc_len:
    :return:
    """
    x_data = []
    for i, len in enumerate(doc_len):
        doc = x[i,:,:].resize_(max_doc_len, max_sen_len, 1).numpy()
        x_data.append(doc[:len, :])

    return torch.tensor(x_data)





