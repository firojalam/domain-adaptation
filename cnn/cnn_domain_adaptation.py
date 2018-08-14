#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:16:11 2017

@author: firojalam
"""

import time

import numpy as np

np.random.seed(1234)

import math as m

import scipy.io
import theano
import theano.tensor as T

from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
import sys
import os
import lasagne
import cPickle
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer, get_output
import data_process
from gensim.models import KeyedVectors
import performance as performance
import pandas as pd
import warnings
import datetime
import optparse
import os, errno


def build_convpool_max(l_in, emb_model, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed    
    embedding_matrix = data_process.prepare_embedding(word_index, emb_model, MAX_NB_WORDS, EMBEDDING_DIM)
    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    print (nb_words)
    embedding_layer = lasagne.layers.EmbeddingLayer(l_in, input_size=nb_words, output_size=EMBEDDING_DIM,
                                                    W=embedding_matrix)  # embedding_matrix
    # embedding_layer.params[embedding_layer.W].remove('trainable')
    # embedding_layer = DimshuffleLayer(embedding_layer, (0, 2, 1))
    # output = get_output(embedding_layer, x)

    convnets = []  # models to be merged
    filter_window_sizes = [2, 3, 4]
    num_filters = [100, 150, 200]
    for filter_len, nb_filter in zip(filter_window_sizes, num_filters):
        conv = Conv1DLayer(embedding_layer, nb_filter, filter_len, stride=1, pad='valid',
                           nonlinearity=lasagne.nonlinearities.rectify)
        conv = lasagne.layers.MaxPool1DLayer(conv, pool_size=filter_len)
        conv = lasagne.layers.FlattenLayer(conv)
        dense = lasagne.layers.DenseLayer(conv, nb_filter, nonlinearity=lasagne.nonlinearities.rectify)
        convnets.append(dense)
    print ("Conv done")
    convpool = lasagne.layers.ConcatLayer(convnets, axis=1)
    #    convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    # convpool = lasagne.layers.dropout(convpool, p=.02)
    # convpool = DenseLayer(convpool,num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

    return convpool


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    # assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def store_params(network, model_file):
    """serialize the model parameters in self.model_file.
    """

    fout = open(model_file, 'w')
    # params = lasagne.layers.get_all_param_values(network, trainable=False)
    netInfo = {'network': network, 'params': lasagne.layers.get_all_param_values(network)}
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    cPickle.dump(netInfo, fout, cPickle.HIGHEST_PROTOCOL)
    fout.close()


def load_params(model_file):
    """load the model parameters from self.model_file.
    """
    print (model_file)

    fin = open(model_file)
    net = cPickle.load(fin)
    all_params = net['params']
    network = net['network']
    lasagne.layers.set_all_param_values(network, all_params)
    fin.close()
    return network


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
    base = os.path.basename(fileame)
    name = os.path.splitext(base)[0]
    outFile = "tsne/" + name + "_dom_adapt.csv"
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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-m', action="store", dest="w2v_model_file")
    parser.add_option('-d', action="store", dest="domain_file")
    parser.add_option('-T', action="store", dest="domain_test_file")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file = options.val_data
    test_file = options.test_data
    domain_file = options.domain_file
    domain_test_file = options.domain_test_file

    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    batch_size = 256
    nb_classes = 2

    modelFile = options.w2v_model_file #"../w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)
    #    emb_model=""
    delim = "\t"
    data, _, _ = data_process.get_data(train_file, delim)
    ul_data, _, _ = data_process.get_data(domain_file, delim)

    data.extend(ul_data)
    print("Number of inst for vocab: " + str(len(data)))

    word_index, tokenizer = data_process.get_tokenizer(data, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    train_x, train_y, train_le, train_labels, _, _ = data_process.get_dev_data_with_id(train_file, tokenizer,
                                                                                       MAX_SEQUENCE_LENGTH, delim)
    dev_x, dev_y, dev_le, dev_labels, _, _ = data_process.get_dev_data_with_id(dev_file, tokenizer, MAX_SEQUENCE_LENGTH,
                                                                               delim)
    test_x, test_y, test_le, test_labels, _, _ = data_process.get_dev_data_with_id(test_file, tokenizer,
                                                                                   MAX_SEQUENCE_LENGTH, delim)
    domain_x, domain_y, _, _, _, _ = data_process.get_dev_data_with_id(domain_file, tokenizer, MAX_SEQUENCE_LENGTH,
                                                                       delim)
    domain_test_x, domain_test_y, domain_test_le, domain_test_labels, _, domain_ids = data_process.get_dev_data_with_id(
        domain_test_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)

    train_y = train_y.astype('int32')
    dev_y = dev_y.astype('int32')
    domain_y = domain_y.astype('int32')
    print(train_x.shape)
    print(dev_x.shape)
    print(test_x.shape)
    print(domain_x.shape)

    train_input_x = T.imatrix('train_input')
    train_y_var = T.imatrix('train_label')
    domain_y_var = T.imatrix('domain_label')

    train_input_x.reshape((batch_size, MAX_SEQUENCE_LENGTH))
    l_in = lasagne.layers.InputLayer((None, MAX_SEQUENCE_LENGTH), train_input_x)
    model2 = build_convpool_max(l_in, emb_model, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
    output = get_output(model2, train_input_x)
    cnn = theano.function([train_input_x], output)
    train_x = cnn(train_x)
    dev_x = cnn(dev_x)
    test_x = cnn(test_x)
    domain_x = cnn(domain_x)
    domain_test_x = cnn(domain_test_x)

    R, C = train_x.shape
    x_sym = T.matrix('x', dtype='float32')
    l_in = lasagne.layers.InputLayer(shape=(None, C), input_var=x_sym)
    convpool = DenseLayer(l_in, num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    convpool = lasagne.layers.dropout(convpool, p=0.01)

    # source_network = build_convpool_max(l_in,emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    source_network = lasagne.layers.DenseLayer(convpool, num_units=nb_classes,
                                               nonlinearity=lasagne.nonlinearities.softmax)

    # domain_network = build_convpool_max(l_in,emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)
    # domain_network = lasagne.layers.dropout(domain_network, p=0.02)
    domain_network = lasagne.layers.DenseLayer(convpool, num_units=nb_classes,
                                               nonlinearity=lasagne.nonlinearities.softmax)

    source_prediction = lasagne.layers.get_output(source_network)
    domain_prediction = lasagne.layers.get_output(domain_network)

    source_loss = T.mean(lasagne.objectives.categorical_crossentropy(source_prediction, train_y_var))
    source_params = lasagne.layers.get_all_params(source_network, trainable=True)
    # domain_prediction = lasagne.layers.get_output(domain_network)
    domain_loss = T.mean(lasagne.objectives.categorical_crossentropy(domain_prediction, domain_y_var))
    domain_params = lasagne.layers.get_all_params(domain_network, trainable=True)

    common = set(source_params) & set(domain_params)

    val = 1e-8
    lambda_val = theano.shared(lasagne.utils.floatX(val))
    # val = T.mean(source_loss) - (domain_loss)
    # print val
    updates1 = lasagne.updates.adadelta(source_loss - (lambda_val * domain_loss), source_params,
                                        learning_rate=0.1)  # update blue and green part
    updates2 = lasagne.updates.adadelta(domain_loss, list(set(domain_params) - common), learning_rate=1.0)
    updates3 = lasagne.updates.adadelta((-lambda_val * domain_loss), list(common), learning_rate=1.0)
    updates1.update(updates2)
    updates2.update(updates3)

    train1 = theano.function([l_in.input_var, train_y_var, domain_y_var], source_loss, updates=updates1)
    train2 = theano.function([l_in.input_var, domain_y_var], domain_loss, updates=updates2)

    test_prediction = lasagne.layers.get_output(source_network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, train_y_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), train_y_var), dtype=theano.config.floatX)

    ##compilation

    val_fn = theano.function([l_in.input_var, train_y_var], [test_loss, test_acc])
    pred_fn = theano.function([l_in.input_var], test_prediction)
    patience = 20
    best_valid = 0
    best_valid_epoch = 0
    best_weights = None
    train_history = {}
    train_history['valid_loss'] = list()
    train_history['epoch'] = list()
    modeldir = "nepal"
    modeldirpath = "models/" + modeldir + "/"
    if not os.path.exists(modeldirpath):
        os.makedirs(modeldirpath)

    base = os.path.basename(train_file)
    tr_file_name = os.path.splitext(base)[0]
    num_epochs = 150
    base_lambda_val = 1e-4
    decay_lambda_val = 0.8

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()

        val = base_lambda_val * (decay_lambda_val ** epoch)
        # val=lambda_val.get_value() * 0.1
        lambda_val.set_value(lasagne.utils.floatX(val))

        for batch in iterate_minibatches(train_x, train_y, batch_size, shuffle=True):
            inputs, targets = batch
            source_tr_y = [0] * len(targets)
            y = np.zeros((len(targets), nb_classes))
            y[np.arange(len(targets)), source_tr_y] = 1
            source_tr_y = y.astype('int32')
            train_err += train1(inputs, targets, source_tr_y)
            train_batches += 1
        #            print ("error1 "+str(train_err))

        for batch in iterate_minibatches(domain_x, domain_y, batch_size, shuffle=True):
            inputs, targets = batch
            domain_tr_y = [1] * len(targets)
            # source_tr_y=[0] * len(targets)
            y = np.zeros((len(targets), nb_classes))
            y[np.arange(len(targets)), domain_tr_y] = 1
            domain_tr_y = y.astype('int32')
            train_err += train2(inputs, domain_tr_y)
            #            print ("error2 "+str(train_err))

            train_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        dev_pred = pred_fn(dev_x)
        dev_pred = np.argmax(dev_pred, axis=1)
        # print (dev_pred)
        # accu,P,R,F1,wAUC,AUC,report =performance.performance_measure_tf(dev_y,pred,dev_le,dev_labels)
        y_prob = [0]
        devoutFile = "dev.txt"
        accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(dev_y, dev_pred, y_prob, dev_le,
                                                                               dev_labels, devoutFile)
        wauc = wAUC * 100
        auc = AUC * 100
        precision = P * 100
        recall = R * 100
        f1_score = F1 * 100
        result = str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
            "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
            "{0:.2f}".format(f1_score)) + "\n"
        print(result)
        # print (report)
        current_valid = train_history['valid_loss'].append(f1_score)
        current_epoch = train_history['epoch'].append(epoch)

        if (epoch % 10 == 0):
            current_valid = train_history['valid_loss'][-1]
            print("f1-cur " + str(current_valid))
            current_epoch = train_history['epoch'][-1]
            if current_valid > best_valid:
                print("f1-best " + str(best_valid))
                best_valid = current_valid
                best_valid_epoch = current_epoch
                model = modeldirpath + tr_file_name + "_" + str(best_valid_epoch) + ".model"
                # remove_prev_model("models/")
                print(model)
                store_params([source_network, domain_network], model)
                best_weights = model
                print(str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(wauc)) + "\t" + str(
                    "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
                    "{0:.2f}".format(f1_score)) + "\n")
                # print report
            elif best_valid_epoch + patience < current_epoch:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {} model {}.".format(best_valid, best_valid_epoch,
                                                                                best_weights))
                break

    source_network, domain_network = load_params(best_weights)
    # source_prediction = lasagne.layers.get_output(source_network)
    # domain_prediction = lasagne.layers.get_output(domain_network)
    # test_prediction = lasagne.layers.get_output(source_network,domain_network, deterministic=True) #lasagne.layers.get_output(source_network)

    print ("domain test")
    y_prob = [0]
    testoutFile = "domain_test.txt"
    test_pred = pred_fn(domain_test_x)
    test_pred = np.argmax(test_pred, axis=1)
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
    print (report)


    print ("source dev")
    dev_pred = pred_fn(dev_x)
    dev_pred = np.argmax(dev_pred, axis=1)
    y_prob = [0]
    devoutFile = "dev.txt"
    accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(dev_y, dev_pred, y_prob, dev_le, dev_labels,
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
    print (report)

    print ("source test")
    y_prob = [0]
    testoutFile = "test.txt"
    test_pred = pred_fn(test_x)
    test_pred = np.argmax(test_pred, axis=1)
    accu, P, R, F1, wAUC, AUC, report = performance.performance_measure_tf(test_y, test_pred, y_prob, test_le,
                                                                           test_labels, devoutFile)
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
    base = os.path.basename(domain_test_file)
    basename = os.path.splitext(base)[0]
    dirname = os.path.dirname(domain_test_file)
    fileName = dirname + "/" + basename + "_domain_pred.csv"
    predictions(domain_test_y, test_pred, domain_test_le, fileName, domain_ids)

    output = get_output(convpool, x_sym)
    out_layer_data = theano.function([x_sym], output)

    data = []
    Y = []

    train_x_last_layer = out_layer_data(train_x)
    print (train_x_last_layer.shape)
    source_tr_y = [0] * len(train_x_last_layer)
    domain_last_l = out_layer_data(domain_test_x)
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
    print ("time taken:")
    print(b - a)
