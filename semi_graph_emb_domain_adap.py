#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sun Apr  2 10:33:22 2017

@author: firojalam
"""
import numpy as np
import sys

from cnn import data_process as data_process
# from cnn import cnn_filter as cnn_filter
from cnn import average_vector as average_vector
from cnn import cnn_feat_extract as cnn_feat_extract
from gensim.models import KeyedVectors
import graph_knn_train
import os
import random
import warnings
import datetime
import optparse
import os, errno

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_s_data")
    parser.add_option('-v', action="store", dest="unlabeled_s_data")
    parser.add_option('-t', action="store", dest="train_t_data")
    parser.add_option('-o', action="store", dest="output_file")
    parser.add_option('-m', action="store", dest="unlabeled_t_data")
    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_source_file = options.train_s_data
    unlabeled_source_file = options.unlabeled_s_data
    tagget_train_file = options.train_t_data
    tagget_unlabeled_file = options.unlabeled_t_data

    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    model_file = options.w2v_model_file  # "../w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(model_file, binary=False)

    total_train_inst = sum(1 for line in open(train_source_file, 'rU')) - 1
    delim = "\t"

    l_source_data, l_source_labels = data_process.get_train_text_data(train_source_file, delim)
    ul_source_data, ul_source_lab, _ = data_process.get_data(unlabeled_source_file, delim)
    l_target_data, l_target_labels = data_process.get_train_text_data(train_source_file, delim)
    ul_target_data, ul_target_lab, _ = data_process.get_data(tagget_unlabeled_file, delim)

    # trOutFile = cnn_feat_extract.writeTxtFile(np.array(l_labels),np.array(l_data),train_source_file,0,"txt")
    R = len(ul_source_data)
    if (R > 50000):
        R = 50000
    random.seed(4)
    random.shuffle(ul_source_data)
    ul_source_data = ul_source_data[0:R]
    random.shuffle(ul_source_lab)
    ul_source_lab = ul_source_lab[0:R]

    R = len(l_target_data)
    if (R > 50000):
        R = 50000
    random.shuffle(ul_target_data)
    ul_target_data = ul_target_data[0:R]
    random.shuffle(ul_target_lab)
    ul_target_lab = ul_target_lab[0:R]

    all_x = []
    all_x.extend(l_source_data)
    all_x.extend(ul_source_data)
    all_x.extend(l_target_data)
    all_x.extend(ul_target_data)

    all_y = []
    all_y.extend(l_source_labels)
    all_y.extend(ul_source_lab)
    all_y.extend(l_target_labels)
    all_y.extend(ul_target_lab)

    print("Train: " + str(len(all_x)))
    allxOutFile = cnn_feat_extract.writeTxtFile(np.array(all_y), np.array(all_x), train_source_file, 0, "all")

    word_index, tokenizer = data_process.get_tokenizer(all_x, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    all_x, _, ule, Ulabels = data_process.get_train_vector(all_x, all_y, tokenizer, MAX_SEQUENCE_LENGTH)

    all_x_feat = average_vector.get_avg_feature_vecs(all_x, emb_model, EMBEDDING_DIM)

    dirname = os.path.dirname(train_source_file)
    base = os.path.basename(train_source_file)
    name = os.path.splitext(base)[0]
    knnModelFile = dirname + "/" + name + "_allfeat.knn.model"
    graphFile = dirname + "/" + name + "_graph.csv"

    graph_knn_train.create_NN_model(all_x_feat, knnModelFile, graphFile)

    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)
