#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:45:26 2017

@author: firojalam
"""

from __future__ import division, print_function, absolute_import

import numpy as np

from sklearn import metrics
import sys
import os
import sklearn.metrics as metrics
from sklearn import preprocessing
import pandas as pd
import performance
import data_process
import cnn_filter
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint
import performance as performance
from gensim.models import KeyedVectors
import warnings
import datetime
import optparse
import os, errno

# for reproducibility
seed = 1337
np.random.seed(seed)


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

    y1 = np.array([np.arange(2)[l == 1][0] for l in y])
    n_classes = y1.max() + 1
    n_from_each_class = int(n_labeled / n_classes)
    i_labeled = []
    for c in range(n_classes):
        print(c)
        i = indices[y1 == c][:n_from_each_class]
        i_labeled += list(i)
    x = x[i_labeled]
    y = y[i_labeled]
    print("data: " + str(len(x)))
    print("label: " + str(len(y)))

    return x, y, le, labels


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
    unlabaled_data = options.unlabeled_data

    MAX_SEQUENCE_LENGTH = 45
    MAX_NB_WORDS = 10000
    EMBEDDING_DIM = 300
    batch_size = 32
    nb_classes = 2
    nb_epoch = 100

    modelFile = options.w2v_model_file #"../w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)

    delim = "\t"
    train_xdata, train_y, le, labels, word_index, tokenizer = data_process.getTrData(train_file, MAX_NB_WORDS,
                                                                                     MAX_SEQUENCE_LENGTH, delim)
    dev_x, dev_y, dev_le, dev_labels, _ = data_process.getDevData2(dev_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)
    test_x, test_y, test_le, test_labels, _ = data_process.getDevData2(test_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)

    ublabelled_X, _, _, _, _ = data_process.getDevData2(unlabaled_data, tokenizer, MAX_SEQUENCE_LENGTH, delim)

    print(train_xdata.shape)
    print(dev_x.shape)
    print(test_x.shape)
    ##### call backs for early stoping and saving the model
    callbacks = callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    best_model_path = "models/weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [callbacks, checkpoint]

    model = cnn_filter.text_cnn(emb_model, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
    model.compile('rmsprop', 'mse')
    print(model.layers[-1].output_shape)
    train_x = model.predict(train_xdata)
    dev_x = model.predict(dev_x)
    test_x = model.predict(test_x)
    ublabelled_X = model.predict(ublabelled_X)
    R, C = train_x.shape

    model = Sequential()
    model.add(Dense(512, batch_input_shape=(None, C)))
    model.add(Dropout(0.02))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    model.fit([train_x], train_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=([dev_x], dev_y),
              callbacks=callbacks_list)

    ########## load the best model and predict
    # load weights
    # model=cnn_filter.text_cnn(emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)

    model = Sequential()
    model.add(Dense(512, batch_input_shape=(None, C)))
    model.add(Dropout(0.02))
    model.add(Dense(nb_classes, activation='softmax'))
    # load weights
    model.load_weights(best_model_path)
    # Compile model (required to make predictions) 
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    dev_pred = model.predict_classes([dev_x], batch_size=batch_size, verbose=1)
    test_pred = model.predict_classes([test_x], batch_size=batch_size, verbose=1)

    ######Dev    
    accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(dev_y, dev_pred, [], dev_le, labels, dev_file)
    wauc = wAUC * 100
    auc = AUC * 100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    print(str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score)) + "\n")
    print(report)

    ######Test
    accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(test_y, test_pred, [], le, labels, test_file)
    wauc = wAUC * 100
    auc = AUC * 100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    print(str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score)) + "\n")
    print(report)

    probability = model.predict_proba(ublabelled_X, batch_size=batch_size, verbose=1)
    ul_pred_val = model.predict_classes([ublabelled_X], batch_size=batch_size, verbose=1)
    ul_pred = le.inverse_transform(ul_pred_val)
    #    fout = open("ul_pred1.txt", 'w')
    delim = "\t"
    train_xdata, lab_tr_x, ids = data_process.getData(train_file, delim)
    uData, uLab, uIds = data_process.getData(unlabaled_data, delim)
    ul_pred_data = []
    ul_pred_labels = []
    ul_pred_ids = []
    for pred, prob, val, udata, id in zip(ul_pred, probability, ul_pred_val, uData, uIds):
        class_prob = prob[val]
        if (class_prob >= 0.75):
            ul_pred_data.append(udata)
            ul_pred_labels.append(pred)
            ul_pred_ids.append(id)

    base = os.path.basename(train_file)
    basename = os.path.splitext(base)[0]
    dirname = os.path.dirname(train_file)
    fileName = dirname + "/" + basename + "_ul.csv"
    fout = open(fileName, 'w')
    fout.write("##id\ttext\tlabel" + "\n");
    # train_xdata=train_xdata.tolist()
    print(str(len(ids)) + " " + str(len(train_xdata)) + " " + str(len(lab_tr_x)))
    for id, x, y in zip(ids, train_xdata, lab_tr_x):
        fout.write(str(id) + "\t" + str(x) + "\t" + str(y) + "\n");

    print(str(len(ul_pred_ids)) + " " + str(len(ul_pred_data)) + " " + str(len(ul_pred_labels)))
    for id, x, y in zip(ul_pred_ids, ul_pred_data, ul_pred_labels):
        fout.write(str(id) + "\t" + str(x) + "\t" + str(y) + "\n");
    fout.close()
    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)
