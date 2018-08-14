#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sun Apr  2 10:33:22 2017

@author: firojalam
"""
import numpy as np
import sys

from cnn import data_process as data_process
#from cnn import cnn_filter as cnn_filter
from cnn import average_vector as average_vector
from cnn import cnn_feat_extract as cnn_feat_extract
from gensim.models import KeyedVectors
import graph_knn_train
import os
from cnn import cnn_filter as cnn_filter
import warnings
import datetime
import optparse
import os, errno

if __name__ == '__main__':    
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-o', action="store", dest="output_file")
    parser.add_option('-u', action="store", dest="unlabelled_file")
    parser.add_option('-m', action="store", dest="w2v_model_file")
    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file=options.val_data
    test_file=options.test_data
    unlabeled_file=options.unlabelled_file
    output_file=options.output_file


    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300

    model_file = options.w2v_model_file
    emb_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    
    #emb_model =""
    total_train_inst = sum(1 for line in open(train_file, 'rU')) - 1
    outFile=open(output_file, "w")
    inst_vec = [100,500,1000,2000,total_train_inst]
    for numInst in inst_vec:
        delim="\t"
        if(total_train_inst!=numInst):            
            l_data,l_labels,U_from_L_data,U_from_L_label = data_process.shuffle_train_data(train_file, numInst, delim)
            l_labels=np.array(l_labels)
            l_data=np.array(l_data)            
            ul_data,ul_lab,_=data_process.get_data(unlabeled_file, delim)
            U_from_L_data=U_from_L_data.tolist()
            U_from_L_label=U_from_L_label.tolist()
            U_from_L_data.extend(ul_data)
            U_from_L_label.extend(ul_lab)
            ul_data=U_from_L_data
            ul_lab=U_from_L_label
        else:
            l_data,l_labels=data_process.get_train_text_data(train_file, delim)
            ul_data,ul_lab,_=data_process.get_data(unlabeled_file, delim)
            
        trOutFile = cnn_feat_extract.writeTxtFile(np.array(l_labels), np.array(l_data), train_file, numInst, "txt-cnn")
            
        R=len(ul_data)
        if(R>50000):
            R=50000
        ul_data=ul_data[0:R] 
        ul_lab=ul_lab[0:R] 
        
        all_x=[]
        all_x.extend(l_data)
        all_x.extend(ul_data)

        all_y=[]
        all_y.extend(l_labels)
        all_y.extend(ul_lab)        

        print("Train: "+str(len(all_x)))
        allxOutFile = cnn_feat_extract.writeTxtFile(np.array(all_y), np.array(all_x), train_file, numInst, "all-cnn")
        
        word_index,tokenizer=data_process.get_tokenizer(all_x, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH) 
        train_x,train_y,train_le,train_labels=data_process.get_train_vector(l_data, l_labels, tokenizer, MAX_SEQUENCE_LENGTH)
        all_x,all_y,ule,Ulabels=data_process.get_train_vector(all_x, all_y, tokenizer, MAX_SEQUENCE_LENGTH)                        
        
        
        model = cnn_filter.text_cnn(emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)        
        model.compile('rmsprop', 'mse')    
        print(model.layers[-1].output_shape)          
        
        #all_x_feat = average_vector.get_avg_feature_vecs(all_x,emb_model,EMBEDDING_DIM)
        all_x_feat=model.predict(all_x)
        
        dirname=os.path.dirname(train_file)
        base=os.path.basename(train_file)
        name=os.path.splitext(base)[0]
        knnModelFile=dirname+"/"+name+"_allfeat.knn.model"
        graphFile=dirname+"/"+name+"_graph_"+str(numInst)+".csv"
        
        graph_knn_train.create_NN_model(all_x_feat, knnModelFile, graphFile)
        outFile.write(os.path.basename(trOutFile) +" " + os.path.basename(dev_file) + " " + os.path.basename(test_file) + " " + os.path.basename(graphFile) + " " + os.path.basename(allxOutFile) + "\n")
    outFile.close()
