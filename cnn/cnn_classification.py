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
from gensim.models import KeyedVectors
import warnings
import datetime
import optparse
import os, errno

# for reproducibility
seed = 1337
np.random.seed(seed)


    
if __name__ == '__main__':    
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-m', action="store", dest="w2v_model_file")
    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)


    train_file = options.train_data
    dev_file=options.val_data
    test_file=options.test_data

    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    batch_size=32

    nb_epoch=500
    model_file= options.w2v_model_file #"../w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(model_file, binary=False)

    delim="\t"
    train_x,train_y,train_le,labels,word_index,tokenizer=data_process.getTrData(train_file, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim)
    dev_x,dev_y,Dle,Dlabels,_=data_process.getDevData2(dev_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)
    test_x,test_y,Tle,Tlabels,_=data_process.getDevData2(test_file, tokenizer, MAX_SEQUENCE_LENGTH, delim)
    
    print(train_x.shape)
    print(dev_x.shape)    
    print(test_x.shape)    

    ##### call backs for early stoping and saving the model
    callbacks = callbacks.EarlyStopping(monitor='val_acc',patience=25,verbose=0, mode='max')
    best_model_path="models/weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [callbacks,checkpoint]
    
    model=cnn_filter.text_cnn(emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)    
    
    model.compile('rmsprop', 'mse')
    print(model.layers[-1].output_shape)  
    train_x = model.predict(train_x)
    dev_x = model.predict(dev_x)
    test_x = model.predict(test_x)    
    R,C=train_x.shape
    
    y_true=np.argmax(train_y, axis = 1)
    y_true=train_le.inverse_transform(y_true)
    nb_classes=len(set(y_true.tolist()))
    
    #    
    model=Sequential()        
    model.add(Dense(512, batch_input_shape=(None,C)))
    model.add(Activation('relu'))
    model.add(Dropout(0.02))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])
    model.fit([train_x], train_y, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=([dev_x], dev_y),callbacks=callbacks_list)

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
    
    dev_pred=model.predict_classes([dev_x], batch_size=batch_size, verbose=1)
    test_pred=model.predict_classes([test_x], batch_size=batch_size, verbose=1)
    
    ######Dev    
    y_prob = model.predict_proba(dev_x,batch_size=batch_size, verbose=1)
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(dev_y, dev_pred, y_prob, Dle, labels, dev_file)
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    print(str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(wauc))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n")
    print (report)
    
    
    ######Test
    y_prob = model.predict_proba(test_x,batch_size=batch_size, verbose=1)
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(test_y, test_pred, y_prob, Tle, labels, test_file)
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    print(str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(wauc))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n")    
    print (report)

    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)
