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
from cnn import cnn_feat_extract as cnn_feat_extract
from gensim.models import KeyedVectors
import graph_knn_train
import os



if __name__ == '__main__':    
    # Read train-set data
    trainFile=sys.argv[1]
    devFile=sys.argv[2]    
    tstFile=sys.argv[3]    
    ulFile=sys.argv[4]    
    fileList=sys.argv[5]    

    MAX_SEQUENCE_LENGTH = 20
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    #model_file="/Users/firojalam/QCRI/w2v/crisis_tweets_w2v_model/model/crisis_word_vector.txt"
    modelFile="/export/home/fialam/crisis_semi_supervised/crisis-tweets/model/crisis_word_vector.txt"    
    emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)
    
    #emb_model =""
    total_train_inst = sum(1 for line in open(trainFile, 'rU'))-1
    outFile=open(fileList,"w")
    inst_vec = [total_train_inst]
    for numInst in inst_vec:
        delim="\t"
        if(total_train_inst!=numInst):            
            l_data,l_labels,U_from_L_data,U_from_L_label = data_process.shuffle_train_data(trainFile, numInst, delim)
            l_labels=np.array(l_labels)
            l_data=np.array(l_data)            
            ul_data,ul_lab,_=data_process.get_data(ulFile, delim)
            U_from_L_data=U_from_L_data.tolist()
            U_from_L_label=U_from_L_label.tolist()
            U_from_L_data.extend(ul_data)
            U_from_L_label.extend(ul_lab)
            ul_data=U_from_L_data
            ul_lab=U_from_L_label
        else:
            l_data,l_labels=data_process.get_train_text_data(trainFile, delim) 
            ul_data,ul_lab,_=data_process.get_data(ulFile, delim)
            
        trOutFile = cnn_feat_extract.writeTxtFile(np.array(l_labels),np.array(l_data),trainFile,numInst,"txt") 
            
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
        allxOutFile = cnn_feat_extract.writeTxtFile(np.array(all_y),np.array(all_x),trainFile,numInst,"all")
        
        word_index,tokenizer=data_process.get_tokenizer(all_x, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH) 
        train_x,train_y,train_le,train_labels=data_process.get_train_vector(l_data, l_labels, tokenizer, MAX_SEQUENCE_LENGTH)
        all_x,all_y,ule,Ulabels=data_process.get_train_vector(all_x, all_y, tokenizer, MAX_SEQUENCE_LENGTH)                        
        
        model = cnn_filter.text_cnn(emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)        
        model.compile('rmsprop', 'mse')    
        print(model.layers[-1].output_shape)          
        all_x_feat = model.predict(all_x)
        dirname=os.path.dirname(trainFile)
        base=os.path.basename(trainFile)
        name=os.path.splitext(base)[0]
        knnModelFile=dirname+"/"+name+"_allfeat.knn.model"
        graphFile=dirname+"/"+name+"_graph_"+str(numInst)+".csv"
        
        graph_knn_train.create_NN_model(all_x_feat, knnModelFile, graphFile)
        outFile.write(os.path.basename(trOutFile)+" "+os.path.basename(devFile)+" "+os.path.basename(tstFile)+" "+os.path.basename(graphFile)+" "+os.path.basename(allxOutFile)+"\n")
    outFile.close()
