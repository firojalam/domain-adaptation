#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:45:26 2017

@author: firojalam
"""
from __future__ import division, print_function, absolute_import

import numpy as np

seed = 1337
np.random.seed(seed)

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
from gensim.models import KeyedVectors
# for reproducibility
from keras.models import Model
import warnings
import datetime
import optparse
import os, errno


def predictions(ture, pred, le, outfile, ids):
    y_true = np.argmax(ture, axis=1)
    # y_pred=np.argmax(pred, axis = 1)
    y_pred = le.inverse_transform(pred)
    y_true = le.inverse_transform(y_true)

    base = os.path.basename(outfile)
    fname = os.path.splitext(base)[0]
    pred_file = "labeled/" + fname + "_pred.txt"
    fopen = open(pred_file, "w");
    fopen.write("##Ref.\tPrediction\n")
    for id1, ref, pred in zip(ids, y_true, y_pred):
        fopen.write(str(id1) + "\t" + str(ref) + "\t" + str(pred) + "\t1.0\n")
    fopen.close


def writeDataFile(Y, data, fileame):
    # dirname=os.path.dirname(fileame)
    base = os.path.basename(fileame)
    name = os.path.splitext(base)[0]
    # out_file=dirname+"/"+name+"_"+featTxt+".csv"
    outFile = "tsne/" + name + "_crossevent.csv"
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


def checkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def checkModel(filename):
    if os.path.exists(filename):
        return True
    return False


def modelPath(modeldir="cross_event"):
    modeldirpath = "models/" + modeldir + "/"
    modeldirpath = checkdir(modeldirpath)
    return modeldirpath;


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-m', action="store", dest="w2v_model_file")
    parser.add_option('-d', action="store", dest="domain_test_file")
    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file = options.val_data
    test_file = options.test_data
    domain_test_file = options.domain_test_file

    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    batch_size = 32

    nb_epoch = 500
    modelFile = options.w2v_model_file  # "../w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)

    delim = "\t"
    train_x, train_y, train_le, labels, word_index, tokenizer = data_process.getTrData(train_file, MAX_NB_WORDS,
                                                                                       MAX_SEQUENCE_LENGTH, delim)
    dev_x, dev_y, Dle, Dlabels, _ = data_process.getDevData2(dev_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)
    test_x, test_y, Tle, Tlabels, _ = data_process.getDevData2(test_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)
    domain_test_x, domain_test_y, domain_test_le, domain_test_labels, _, domain_ids = data_process.getDevDataWithID(
        domain_test_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)

    print(train_x.shape)
    print(dev_x.shape)
    print(test_x.shape)

    ##### call backs for early stoping and saving the model
    callbacks = callbacks.EarlyStopping(monitor='val_acc', patience=25, verbose=0, mode='max')
    best_model_path = "models/weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [callbacks, checkpoint]

    model = cnn_filter.text_cnn(emb_model, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)

    model.compile('rmsprop', 'mse')
    print(model.layers[-1].output_shape)
    train_x = model.predict(train_x)
    dev_x = model.predict(dev_x)
    test_x = model.predict(test_x)
    domain_test_x = model.predict(domain_test_x)
    R, C = train_x.shape

    y_true = np.argmax(train_y, axis=1)
    y_true = train_le.inverse_transform(y_true)
    nb_classes = len(set(y_true.tolist()))

    #    
    model = Sequential()
    model.add(Dense(512, batch_input_shape=(None, C), name="dense_out"))
    model.add(Activation('relu'))
    model.add(Dropout(0.01))
    model.add(Dense(nb_classes, activation='softmax'))

    base = os.path.basename(train_file)
    basename = os.path.splitext(base)[0]
    modeldirpath = modelPath() + "/" + basename + ".model.weights"
    if (checkModel(modeldirpath)):
        model.load_weights(modeldirpath)
    else:
        model.save_weights(modeldirpath)

    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    model.fit([train_x], train_y, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=([dev_x], dev_y),
              callbacks=callbacks_list)

    ########## load the best model and predict
    # load weights
    #    model=cnn_filter.text_cnn(emb_file,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    #
    #    model.add(Dropout(0.2))
    #    model.add(Dense(nb_classes, activation='softmax'))
    #    # load weights
    #    model.load_weights(best_model_path)
    #    # Compile model (required to make predictions)
    #    model.compile(loss='binary_crossentropy',optimizer='Adadelta',metrics=['accuracy'])

    dev_pred = model.predict_classes([dev_x], batch_size=batch_size, verbose=1)
    test_pred = model.predict_classes([test_x], batch_size=batch_size, verbose=1)

    ######Dev    
    print("source dev.......")
    y_prob = model.predict_proba(dev_x, batch_size=batch_size, verbose=1)
    accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(dev_y, dev_pred, y_prob, Dle, labels,
                                                                           dev_file)
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
    print("source test.......")
    y_prob = model.predict_proba(test_x, batch_size=batch_size, verbose=1)
    accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(test_y, test_pred, y_prob, Tle, labels,
                                                                           test_file)
    wauc = wAUC * 100
    auc = AUC * 100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    print(str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score)) + "\n")
    print(report)
    print("domain test.......")
    y_prob = [0]
    devoutFile = "domain-test.txt"
    test_pred = model.predict_classes(domain_test_x)
    # test_pred=np.argmax(test_pred, axis = 1)
    accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(domain_test_y, test_pred, y_prob,
                                                                           domain_test_le, domain_test_labels,
                                                                           devoutFile)
    wauc = wAUC * 100
    auc = AUC * 100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    result = str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score)) + "\n"
    print(result)
    print(report)
    base = os.path.basename(domain_test_file)
    basename = os.path.splitext(base)[0]
    dirname = os.path.dirname(domain_test_file)
    fileName = dirname + "/" + basename + "_base_pred.csv"
    predictions(domain_test_y, test_pred, domain_test_le, fileName, domain_ids)
    layer_name = 'dense_out'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    train_x_last_layer = intermediate_layer_model.predict(train_x)

    data = []
    Y = []

    domain_x_last_layer = intermediate_layer_model.predict(train_x)
    print(domain_x_last_layer.shape)
    source_tr_y = [0] * len(train_x_last_layer)
    domain_last_l = intermediate_layer_model.predict(domain_test_x)
    domain_tr_y = [1] * len(domain_last_l)

    data.extend(train_x_last_layer)
    data.extend(domain_last_l)
    Y.extend(source_tr_y)
    Y.extend(domain_tr_y)
    Y = np.array(Y, 'int32')
    data = np.array(data)
    domain_tr_y = domain_last_l.astype('int32')
    writeDataFile(Y, data, train_file)
    b = datetime.datetime.now().replace(microsecond=0)
    print("time taken:")
    print(b - a)
