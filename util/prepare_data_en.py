import numpy as np
import random
import codecs
import pickle as pk
import json
path = '../data/'
###train
# max_doc_len = 45
# max_sen_len = 55

###test

# max_sen_len = 130
# max_doc_len = 40

###将train和test合并起来选取一个最大值
max_sen_len = 130
max_doc_len = 45
###主要是将english 文本处理为和中文文本相同的格式的

def prepare_data(file_in_path, file_out_path):

    f_in = open(file_in_path, 'r', errors='ignore')
    line = f_in.readline()
    line = f_in.readline()

    max_sen_len = -1
    max_sen_len_doc_id = -1
    max_doc_len = -1

    f_out = open(file_out_path, 'w')

    while line:
        line_info = line.split(',')
        doc_id = line_info[0]
        doc_text = line_info[1].strip()
        keywords = line_info[2]
        clause_pos = line_info[3]
        label = line_info[4]

        clauses = doc_text.split('\x01')
        clause_poses = clause_pos.split(' ')

        if len(clause_poses) > max_doc_len:
            max_doc_len = len(clause_poses)

        for c in clauses:
            c_len = len(c.strip().split(' '))
            if c_len > max_sen_len:
                max_sen_len = c_len
                max_sen_len_doc_id = doc_id
        if '0' in clause_poses:
            emotion_pos = clause_poses.index('0')
        else:
            emotion_pos = 0
        emotion_labels = ['0']*len(clause_poses)
        emotion_labels[emotion_pos] = '1'
        labels = label.split(' ')
        # if doc_id == '167':
        #     print(doc_id)

        for c_indx, clause in enumerate(zip(emotion_labels, clause_poses, labels, clauses)):
            row = doc_id + ',' + str(c_indx) + ',' + keywords + ',' + ','.join(clause)
            f_out.write(row)
            f_out.write('\n')
        line = f_in.readline()

    f_out.close()
    print('max_sen_len', max_sen_len)
    print('max_doc_len', max_doc_len)
    print('max_sen_len_doc_id', max_sen_len_doc_id)


def load_w2v(embedding_dim, file_path, embedding_path):
    print('\nload embedding...')
    words = []
    inputFile1 = codecs.open(file_path, 'r', 'GBK', errors='ignore')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())
    inputFile1.close()
    words = set(words)  # 所有不重复词的集合
    word_idx = dict((c, k + 1) for k, c in enumerate(words))
    idx_word = dict((k+1, c) for k, c in enumerate(words))

    w2v = {}
    inputFile2 = codecs.open(embedding_path, 'r', 'utf-8')
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd
    inputFile2.close()
    embedding = [list(np.zeros(embedding_dim))] #0 padding on 0
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)  # 从均匀分布[-0.1,0.1]中随机取
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(
        embedding_path, len(words), hit))

    embedding = np.array(embedding)

    pk.dump(embedding, open(path + 'all_embedding_en.txt', 'wb'))
    pk.dump(json.dumps(idx_word), open(path + 'all_idx_word_en.txt', 'wb'))


    print("embedding.shape: {}".format(
        embedding.shape))
    print("load embedding done!\n")
    return word_idx

def load_data(input_file, word_idx, max_doc_len=max_doc_len, max_sen_len=max_sen_len):
    print('load data...')
    doc_id, emotion_id, x, y, sen_len, doc_len = [], [], [], [], [], []

    y_clause, clause_all, tmp_clause_len =\
        np.zeros((max_doc_len, 2)), [], []
    next_ID = 1
    outputFile3 = codecs.open(input_file, 'r', 'GBK', errors='ignore')
    n_clause, emotion_clause, cause_clause, emotion_cause_clause, n_cut = [0] * 5
    doc_id.append(0)

    for index, line in enumerate(outputFile3.readlines()):
        n_clause += 1
        line = line.strip().split(',')
        print(line[0])
        doc_ID, clause_idx, key_word, emotion_label, rp, cause_label, words =\
            int(line[0]), int(line[1]), line[2], int(line[3]), int(line[4]), int(line[5]), line[-1]
        if next_ID == doc_ID:  # 数据文件末尾加了一个冗余的文档，会被丢弃
            doc_len.append(len(clause_all))
            doc_id.append(doc_ID)

            for j in range(max_doc_len - len(clause_all)):
                clause_all.append(np.zeros((max_sen_len,)))
                tmp_clause_len.append(0)
            x.append(clause_all)
            y.append(y_clause)
            sen_len.append(tmp_clause_len)
            y_clause, clause_all, tmp_clause_len = \
                np.zeros((max_doc_len, 2)), [], []
            next_ID = doc_ID + 1

        clause = [0] * max_sen_len
        for i, word in enumerate(words.split()):
            if word in word_idx.keys():
                clause[i] = int(word_idx[word])
            else:
                clause[i] = 0
        clause_all.append(np.array(clause))
        tmp_clause_len.append(len(words.split()))
        if cause_label == 1:
            emotion_cause_clause += 1
            y_clause[clause_idx] = [0,1]
        else:
            y_clause[clause_idx] = [1,0]
        if emotion_label == 1:
            emotion_id.append(clause_idx+1)


    outputFile3.close()
    x = np.array(x)
    y = np.array(y)
    sen_len = np.array(sen_len)
    doc_len = np.array(doc_len)
    # x, y, sen_len, doc_len = map(np.array, [x, y, sen_len, doc_len])
    doc_id = np.array(doc_id)[0:-1]
    emotion_id = np.array(emotion_id)
    max_len = np.argmax(doc_len)
    print(max_len)
    pk.dump(x, open(path + 'test_x_en.txt', 'wb'))
    pk.dump(y, open(path + 'test_y_en.txt', 'wb'))
    pk.dump(sen_len, open(path + 'test_sen_len_en.txt', 'wb'))
    pk.dump(doc_len, open(path + 'test_doc_len_en.txt', 'wb'))
    pk.dump(doc_id, open(path + 'test_doc_id_en.txt', 'wb'))
    pk.dump(emotion_id, open(path + 'test_emotion_id_en.txt', 'wb'))


    print('doc_id.shape {}\nemotion_id.shape{}\nx.shape {} \ny.shape {}\nsen_len.shape {} \ndoc_len.shape {}\n'.format(
        doc_id.shape, emotion_id.shape, x.shape, y.shape, sen_len.shape, doc_len.shape
    ))
    print('n_clause {}, emotion_clause {}, cause_clause {}, emotion_cause_clause {}, n_cut {}'.format(
        n_clause, emotion_clause, cause_clause, emotion_cause_clause, n_cut))
    print('load data done!\n')
    return x, y, sen_len, doc_len

# file_in_path = '../data/ntcir_eng_train.csv'
# file_out_path = '../data/ntcir_eng_train_pre.txt'
# prepare_data(file_in_path, file_out_path)
word_dict = load_w2v(300, path + 'ntcir_eng_all_pre.txt', path + 'glove.6B.300d.txt')
load_data(path + 'ntcir_eng_test_pre.txt', word_dict)
