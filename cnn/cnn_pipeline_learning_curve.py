#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sun Apr  2 10:33:22 2017

@author: firojalam
"""
import numpy as np
import sys
from cnn import data_process as data_process
from cnn import cnn_filter as cnn_filter
from gensim.models import KeyedVectors
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint
import cnn.performance as performance
import os
import warnings
import datetime
import optparse
import os, errno

seed = 1337
np.random.seed(seed)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-r', action="store", dest="results_file")
    parser.add_option('-m', action="store", dest="w2v_model_file")
    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file = options.val_data
    test_file = options.test_data
    results_file = options.results_file
    out_file = open(results_file, "w")

    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    batch_size = 32
    nb_epoch = 100
    model_file = options.w2v_model_file  # "../w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(model_file, binary=False)

    # emb_model =""
    total_train_inst = sum(1 for line in open(train_file, 'rU')) - 1

    inst_vec = [100, 500, 1000, 2000, total_train_inst]
    for numInst in inst_vec:
        delim = "\t"
        if (total_train_inst != numInst):
            l_data, l_labels, U_from_L_data, U_from_L_label = data_process.shuffle_train_data(train_file, numInst,
                                                                                              delim)
            l_labels = np.array(l_labels)
            l_data = np.array(l_data)

            all_x = []
            all_x.extend(l_data)
            all_y = []
            all_y.extend(l_labels)
            word_index, tokenizer = data_process.get_tokenizer(all_x, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
            train_x, train_y, train_le, train_labels = data_process.get_train_vector(all_x, all_y, tokenizer,
                                                                                     MAX_SEQUENCE_LENGTH)

        else:
            train_x, train_y, train_le, train_labels, word_index, tokenizer = data_process.get_train_data(train_file,
                                                                                                          MAX_NB_WORDS,
                                                                                                          MAX_SEQUENCE_LENGTH,
                                                                                                          delim)

        dev_x, dev_y, dev_le, dev_labels, _ = data_process.get_dev_data2(dev_file, tokenizer, MAX_SEQUENCE_LENGTH,
                                                                         delim)
        test_x, test_y, test_le, test_labels, _ = data_process.get_dev_data2(test_file, tokenizer, MAX_SEQUENCE_LENGTH,
                                                                             delim)
        print("Train: " + str(len(train_x)))

        y_true = np.argmax(train_y, axis=1)
        y_true = train_le.inverse_transform(y_true)
        nb_classes = len(set(y_true.tolist()))
        print ("Number of classes: " + str(nb_classes))

        model = cnn_filter.text_cnn(emb_model, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
        model.compile('rmsprop', 'mse')
        print(model.layers[-1].output_shape)

        train_x = model.predict(train_x)
        dev_x = model.predict(dev_x)
        test_x = model.predict(test_x)

        callback = callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
        best_model_path = "models/weights.best.hdf5"
        checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [callback, checkpoint]
        R, C = train_x.shape
        # model=Sequential()
        model.add(Dense(512))
        model.add(Activation('relu'))
        # model.add(Dropout(0.02))
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
        model.fit([train_x], train_y, batch_size=batch_size, epochs=nb_epoch, verbose=1,
                  validation_data=([dev_x], dev_y), callbacks=callbacks_list)

        dev_pred = model.predict_classes([dev_x], batch_size=batch_size, verbose=1)
        test_pred = model.predict_classes([test_x], batch_size=batch_size, verbose=1)

        ######Dev    
        print("\nNumber of Inst:" + str(numInst))
        y_prob = model.predict_proba(dev_x, batch_size=batch_size, verbose=1)
        accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(dev_y, dev_pred, y_prob, dev_le,
                                                                               dev_labels, dev_file)
        wauc = wAUC * 100
        auc = AUC * 100
        precision = P * 100
        recall = R * 100
        f1_score = F1 * 100
        result = str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
            "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
            "{0:.2f}".format(f1_score)) + "\n"
        print(result)
        print (report)
        out_file.write(str(numInst) + "\t" + dev_file + "\n")
        out_file.write(result)
        out_file.write(report)

        ######Test
        y_prob = model.predict_proba(test_x, batch_size=batch_size, verbose=1)
        accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(test_y, test_pred, y_prob, test_le,
                                                                               test_labels, test_file)
        wauc = wAUC * 100
        auc = AUC * 100
        precision = P * 100
        recall = R * 100
        f1_score = F1 * 100
        result = str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
            "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
            "{0:.2f}".format(f1_score)) + "\n"
        print(result)
        print (report)
        out_file.write(str(numInst) + "\t" + test_file + "\n")
        out_file.write(result)
        out_file.write(report)

    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)
