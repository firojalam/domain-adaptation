#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np

from sklearn import metrics
import sys
import os
from sklearn import preprocessing
import pandas as pd
from cnn import data_process
from cnn import cnn_filter
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint
from gensim.models import KeyedVectors

# for reproducibility
seed = 1337
np.random.seed(seed)
import warnings
import datetime
import optparse
import os, errno


def getData(dataFile):
    """
    Prepare the data
    """
    train = pd.read_csv(dataFile, header=1, delimiter=",")
    R, C = train.shape
    x = train.iloc[:, 0:(C - 2)]
    x = np.array(x.values, dtype=np.float32)
    yL = train.iloc[:, C - 1]
    # print (yL)
    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(yL)
    labels = list(le.classes_)

    # converting to one-hot vector
    label = yL.tolist()

    yC = len(set(label))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    n_labeled = x.shape[0]
    # print(n_labeled)
    indices = np.arange(n_labeled - 1)
    shuffled_indices = np.random.permutation(indices)
    x = x[shuffled_indices]
    y = y[shuffled_indices]
    #    print(y)
    y1 = np.array([np.arange(2)[l == 1][0] for l in y])
    n_classes = y1.max() + 1
    n_from_each_class = int(n_labeled / n_classes)
    i_labeled = []
    # print("class "+str(n_from_each_class))
    # print(y1)
    for c in range(n_classes):
        print(c)
        i = indices[y1 == c][:n_from_each_class]
        i_labeled += list(i)
    x = x[i_labeled]
    y = y[i_labeled]
    print("data: " + str(len(x)))
    print("label: " + str(len(y)))

    # print(y)
    # print(labels)
    return x, y, le, labels


def performance_measure(model, x_tst, y_tst, le, labels):
    numClass = len(labels)
    pred_prob = np.empty((x_tst.shape[0], numClass))
    for ii in range(0, x_tst.shape[0]):
        tvv = np.array(model.predict([x_tst[ii]]))
        pred_prob[ii,] = tvv[0]

    y_true = np.argmax(y_tst, axis=1)
    y_true = le.inverse_transform(y_true)
    lab = list(set(y_true.tolist()))
    lab.sort()
    y_pred = np.argmax(pred_prob, axis=1)
    y_pred = le.inverse_transform(y_pred)
    acc = P = R = F1 = AUC = 0.0
    report = ""
    try:
        acc = metrics.accuracy_score(y_true, y_pred)
        P = metrics.precision_score(y_true, y_pred, average="weighted")
        R = metrics.recall_score(y_true, y_pred, average="weighted")
        F1 = metrics.f1_score(y_true, y_pred, average="weighted")
        AUC = metrics.roc_auc_score(y_true, y_pred, average="weighted")
        report = metrics.classification_report(y_true, y_pred)
    except Exception as e:
        print(e)
        pass
    return acc, P, R, F1, AUC, report


def writeTxtFile(Y, data, fileame, numInst, feat):
    dirname = os.path.dirname(fileame)
    base = os.path.basename(fileame)
    name = os.path.splitext(base)[0]
    outFile = dirname + "/" + name + "_" + feat + "_" + str(numInst) + ".csv"

    featCol = ['text']
    #    feat=np.arange(data.shape[1])
    #    for f in feat:
    #        featCol.append("F"+str(f))
    index = np.arange(data.shape[0])
    print(data.shape)
    df1 = pd.DataFrame(data, index=index, columns=featCol)
    df2 = pd.DataFrame(Y, index=index, columns=["class"])
    output = pd.concat([df1, df2], axis=1)
    output.to_csv(outFile, index=True, quoting=3, sep="\t")
    return outFile


def writeDataFile(Y, data, fileame, numInst, featTxt):
    dirname = os.path.dirname(fileame)
    base = os.path.basename(fileame)
    name = os.path.splitext(base)[0]
    # out_file=dirname+"/"+name+"_"+featTxt+".csv"
    outFile = dirname + "/" + name + "_" + featTxt + "_" + str(numInst) + ".csv"
    print(outFile)
    featCol = []
    feat = np.arange(data.shape[1])
    for f in feat:
        featCol.append("F" + str(f))
    index = np.arange(data.shape[0])

    df1 = pd.DataFrame(data, index=index, columns=featCol)
    df2 = pd.DataFrame(Y, index=index, columns=["class"])
    output = pd.concat([df1, df2], axis=1)
    output.to_csv(outFile, index=False, quoting=3)
    return outFile


def write2File(Y, data, fileame):
    dirname = os.path.dirname(fileame)
    base = os.path.basename(fileame)
    name = os.path.splitext(base)[0]
    outFile = dirname + "/" + name + "_out.csv"

    featCol = []
    feat = np.arange(data.shape[1])
    for f in feat:
        featCol.append("F" + str(f))
    index = np.arange(data.shape[0])

    df1 = pd.DataFrame(data, index=index, columns=featCol)
    df2 = pd.DataFrame(Y, index=index, columns=["class"])
    output = pd.concat([df1, df2], axis=1)
    output.to_csv(outFile, index=False, quoting=3)
    return outFile


def write2FileAllFeat(data, fileame, numInst):
    dirname = os.path.dirname(fileame)
    base = os.path.basename(fileame)
    name = os.path.splitext(base)[0]
    outFile = dirname + "/" + name + "_allfeat_" + str(numInst) + ".csv"

    featCol = []
    feat = np.arange(data.shape[1])
    for f in feat:
        featCol.append("F" + str(f))
    index = np.arange(data.shape[0])

    df1 = pd.DataFrame(data, index=index, columns=featCol)
    # df2 = pd.DataFrame(Y,index=index,columns=["class"])
    # output=pd.concat([df1, df2], axis=1)
    df1.to_csv(outFile, index=False, quoting=3)
    return outFile


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-m', action="store", dest="w2v_model_file")
    parser.add_option('-u', action="store", dest="unlabeled_data")
    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file = options.val_data
    test_file = options.test_data
    unlabeled_file = options.unlabeled_data

    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 10000
    EMBEDDING_DIM = 300
    batch_size = 32
    nb_classes = 2
    nb_epoch = 25
    model_file = options.w2v_model_file

    train_x, train_y, le, labels, word_index, tokenizer = data_process.get_train_data(train_file, MAX_NB_WORDS,
                                                                                      MAX_SEQUENCE_LENGTH)
    dev_x, dev_y, Dle, Dlabels, _ = data_process.get_dev_data(dev_file, tokenizer, MAX_SEQUENCE_LENGTH)
    test_x, test_y, Tle, Tlabels, _ = data_process.get_dev_data(test_file, tokenizer, MAX_SEQUENCE_LENGTH)
    ul_x, ul_y, ule, Ulabels, _ = data_process.get_dev_data(unlabeled_file, tokenizer, MAX_SEQUENCE_LENGTH)

    print(train_x.shape)
    print(dev_x.shape)
    print(test_x.shape)

    emb_model = KeyedVectors.load_word2vec_format(model_file, binary=False)

    model = cnn_filter.text_cnn(emb_model, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
    model.compile('rmsprop', 'mse')
    print(model.layers[-1].output_shape)
    train_x = model.predict(train_x)
    train_y = np.argmax(train_y, axis=1)
    train_y = le.inverse_transform(train_y)
    print(train_y)
    write2File(train_y, train_x, train_file)

    dev_x = model.predict(dev_x)
    dev_y = np.argmax(dev_y, axis=1)
    dev_y = Dle.inverse_transform(dev_y)
    write2File(dev_y, dev_x, dev_file)

    test_x = model.predict(test_x)
    test_y = np.argmax(test_y, axis=1)
    test_y = Tle.inverse_transform(test_y)
    write2File(test_y, test_x, test_file)

    ul_x = model.predict(ul_x)
    ul_y = np.argmax(ul_y, axis=1)
    ul_y = ule.inverse_transform(ul_y)
    write2File(ul_y, ul_x, unlabeled_file)

    all_feat = np.vstack((train_x, ul_x, dev_x, test_x))
    write2FileAllFeat(all_feat, train_file)

    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)
