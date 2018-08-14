# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017

@author: firojalam
"""


import numpy as np
# for reproducibility
seed = 1337
np.random.seed(seed)

#from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys
from sklearn import preprocessing
import pandas as pd
import sklearn.metrics as metrics
import data_process
from keras.models import Sequential
from keras.layers import Convolution1D, GlobalMaxPooling1D
import subprocess
import shlex
from subprocess import Popen, PIPE  
import keras.backend as K
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
from keras.layers import Merge, merge
from keras.layers.normalization import BatchNormalization




def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)

    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    #
    return exitcode, out, err
    
def label_one_hot(yL):
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    return y  
    
def upsampling(train_x,train_y):
    ########## Upsampling    
    y_true=np.argmax(train_y, axis = 1)
    smote = SMOTE(ratio=0.5, kind='borderline1',n_jobs=5)
    X_resampled, y_resampled = smote.fit_sample(train_x,y_true)
  
    ########## Shuffling  
    combined = list(zip(X_resampled, y_resampled))
    random.shuffle(combined)
    X_resampled[:], y_resampled[:] = zip(*combined)
    y_resampled_true=label_one_hot(y_resampled)
    dimension = X_resampled.shape[1]
    y_resampled_true=label_one_hot(y_resampled)
    print(len(X_resampled))
    X_resampled=np.array(X_resampled)
    print(X_resampled.shape)    
    counts = Counter(y_resampled)
    print(counts)   
    return X_resampled, y_resampled_true, dimension
  
def text_cnn(model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH):
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed    
    embedding_matrix=data_process.prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM)    
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    
    
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)    
    ########## CNN: Filtering with Max pooling:
    #nb_filter = 250
    #filter_length = 3
    branches = [] # models to be merged
    filter_window_sizes=[2,3,4]
    num_filters=[100,150,200]
    for filter_len,nb_filter in zip(filter_window_sizes,num_filters):
        branch = Sequential()
        branch.add(embedding_layer)
        branch.add(Conv1D(filters=nb_filter,
                                 kernel_size=int(filter_len),
                                 padding='valid',
                                 activation='relu',
                                 strides=1,
                                 kernel_initializer='glorot_uniform'))
        branch.add(MaxPooling1D(pool_size=filter_len))
        branch.add(Flatten())
        branch.add(Dense(nb_filter,init='uniform', activation='relu'))      
        branches.append(branch)
    model_lex = Sequential()
    model_lex.add(Merge(branches, mode='concat'))      
    return model_lex
#    ########## compile the model:
#    model_lex.compile('rmsprop', 'mse')
#    train_x = model_lex.predict(train_x)
#    dev_x = model_lex.predict(dev_x)
#    test_x = model_lex.predict(test_x)    
#    print(model_lex.layers[-1].output_shape)  
#    return train_x, dev_x, test_x
    

  