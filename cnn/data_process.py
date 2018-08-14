# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017

@author: Firoj Alam
"""


import numpy as np
np.random.seed(1337)  # for reproducibility

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import sys
from sklearn import preprocessing
import pandas as pd
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
 
def get_data(dataFile, delim):
    """
    Prepare the data
    """  
    data=[]
    ids=[]    
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            try:
                line = line.strip()   
                if (line==""):
                    continue         
                #print line                   		
                row=line.split(delim)
                label = row[2]
                txt = row[1].strip()
                if(len(txt)<1):
                    print txt
                    continue
                #txt=aidrtokenize.tokenize(txt)
                #txt=[w for w in txt if w not in stop_words]              
                if(isinstance(txt, str)):                
                    ids.append(row[0])
                    data.append(txt)
                    lab.append(label)
                else:
                    print(txt)
            except Exception as e:
                pass
    return data,lab,ids

def get_train_text_data(dataFile, delim):
    """
    Prepare the data
    """  
    data=[]
    labels=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            label = row[2]
            txt = row[1].strip()
            if(len(txt)<1):
                print txt
                continue
            if(isinstance(txt, str)):                
                data.append(txt)
                labels.append(label)
            else:
                print(txt)
    return data,labels
                
def shuffle_train_data(dataFile, numInst, delim):
    """
    Prepare the data
    """  
    data=[]
    labels=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            label = row[2]
            txt = row[1].strip()
            if(len(txt)<1):
                print txt
                continue
            if(isinstance(txt, str)):                
                data.append(txt)
                labels.append(label)
            else:
                print(txt)
    ##numInst=100# len(data)
    data=np.array(data)        
    labels=np.array(labels)
    indices = np.arange(len(data))
    np.random.seed(1337)
    shuffled_indices = np.random.permutation(indices)
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(labels)
    
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)    
    ynew = np.array([np.arange(yC)[l==1][0] for l in y])
    n_classes = yC
    n_from_each_class =  numInst/ n_classes
    i_labeled = []
    for c in range(n_classes):
        i = indices[ynew==c][:n_from_each_class]
        i_labeled += list(i)
    l_data = data[i_labeled]
    l_labels = labels[i_labeled]
    print("data: "+str(len(l_data)))
    print("label: "+str(len(l_labels)))    
    # Unlabled DataSet        
    U_from_L_indices = [x for x in indices if x not in i_labeled]
    U_from_L_data=data[U_from_L_indices]
    U_from_L_label=labels[U_from_L_indices]        
    print("ul data: "+str(len(U_from_L_data)))
    print("ul label: "+str(len(U_from_L_label)))   
     
    
    return l_data,l_labels,U_from_L_data,U_from_L_label

def get_tokenizer(data, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    """
    Prepare the data
    """  
    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)    
    print('Shape of data tensor:', data.shape)
    return word_index,tokenizer

def get_train_vector(data, lab, tokenizer, MAX_SEQUENCE_LENGTH):
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(lab)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    sequences = tokenizer.texts_to_sequences(data)   
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data,y,le,labels    
    
def get_train_data(dataFile, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """  
#    train = pd.read_csv(dataFile, header=1, delimiter="\t" )  
#    R,C=train.shape
##    print (train)
#    ids=train.iloc[:, 0]
#    ids=ids.values.tolist()    
#    texts=train.iloc[:, 1]
#    texts=texts.values.tolist()

    data=[]
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            label = row[2]
            txt = row[1].strip()
            if(len(txt)<1):
                print txt
                continue
            #txt=aidrtokenize.tokenize(txt)
            #txt=[w for w in txt if w not in stop_words]              
            if(isinstance(txt, str)):                
                data.append(txt)
                lab.append(label)
            else:
                print(txt)
            
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(lab)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    #labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    #print('Shape of label tensor:', labels.shape)    
    #return data,labels,word_index,dim;        
    return data,y,le,labels,word_index,tokenizer

def get_train_data_label_encoder(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """      
    data=[]
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            label = row[2]
            txt = row[1].strip()
            if(len(txt)<1):
                print txt
                continue
            #txt=aidrtokenize.tokenize(txt[1:-1])
            #txt=[w for w in txt if w not in stop_words]              
            data.append(txt)
            lab.append(label)
            
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(lab)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)   
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data,y,le,labels,word_index  
    
def get_dev_data(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):
    """
    Prepare the data
    """      
    data=[]
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            label = row[2]
            txt = row[1].strip()
            if(len(txt)<1):
                print txt
                continue
            #txt=aidrtokenize.tokenize(txt[1:-1])
            #txt=[w for w in txt if w not in stop_words]              
            data.append(txt)
            lab.append(label)
            
    le = train_le #preprocessing.LabelEncoder()
    yL=le.transform(lab)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)   
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data,y,le,labels,word_index    
    
def get_dev_data2(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """      
    data=[]
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            label = row[2]
            txt = row[1].strip()
            if(len(txt)<1):
                print txt
                continue
            #txt=aidrtokenize.tokenize(txt[1:-1])
            #txt=[w for w in txt if w not in stop_words]              
            data.append(txt)
            lab.append(label)
            
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(lab)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)   
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data,y,le,labels,word_index     
    
def get_dev_data_with_id(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """      
    data=[]
    ids=[]    
    lab=[]
    with open(dataFile, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)            
            
            txt = row[1].strip()
            if(len(txt)<1):
                print txt
                continue
            #txt=aidrtokenize.tokenize(txt[1:-1])
            #txt=[w for w in txt if w not in stop_words]              
            label = row[2]
            data.append(txt)
            lab.append(label)
            ids.append(row[0])
            
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(lab)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)   
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data,y,le,labels,word_index,ids      

def load_embedding(fileName):
    print('Indexing word vectors.')    
    embeddings_index = {}    
    f = open(fileName)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()    
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index;

def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
    
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)    
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM),dtype=np.float32)
    print(len(embedding_matrix))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = model[word][0:EMBEDDING_DIM] #embeddings_index.get(word)
            embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
        except KeyError:
            try:
                rng = np.random.RandomState()        	
                embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)
                embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
            except KeyError:    
                continue      
    return embedding_matrix;
    
   
