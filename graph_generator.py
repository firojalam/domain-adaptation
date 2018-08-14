#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 14:04:43 2017

@author: firojalam
"""
import optparse
import datetime
import aidrtokenize;
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from gensim.similarities.docsim import WmdSimilarity
import warnings
import datetime
import optparse
import os, errno

stop_words = stopwords.words('english')

def graph_dist(tweetlist,model,outFile):
    of=open(outFile,"w")   
    model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    #rowVector=[]
    index=0;
    for tweetR in tweetlist:    
        rVec = aidrtokenize.tokenize(tweetR)
        rVec =[w for w in rVec if w not in stop_words]
        colVector=[]
        for tweetC in tweetlist:
            cVec =aidrtokenize.tokenize(tweetC)
            cVec =[w for w in cVec if w not in stop_words]
            distance = model.wmdistance(rVec, cVec)
            colVector.append(distance) 
        vector=str(index)+" "
        for val in colVector:
            vector=vector+str(1-val)+" "
        of.write(vector+"\n")
        #rowVector.append(colVector)   
    of.close()

def graph_sim(tweetlist,model,outFile):
    print("Number of tweets to generate the graph: "+str(len(tweetlist)))
    of=open(outFile,"w")   
    model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    rowVector=[]
    
    for tweetR in tweetlist:    
        rVec = aidrtokenize.tokenize(tweetR)
        rVec =[w for w in rVec if w not in stop_words]
        rowVector.append(rVec)
    instance = WmdSimilarity(rowVector, model, num_best=None)
    #index=0;
    print("Writing into the file....")
    for index,colVector in enumerate(instance):
        vector=""
        for i,val in enumerate(colVector):
            if(index != i and val>=0.3):
                #print str(index)+" != "+str(i)
                vector=vector+str(i)+" "
        of.write(str(index)+" "+vector.strip()+"\n")
    of.close()
    
if __name__ == '__main__':
    a = datetime.datetime.now().replace(microsecond=0) 
    modelFile="/Users/firojalam/QCRI/w2v/GoogleNews-vectors-negative300.bin"
    
    
    model = Word2Vec.load_word2vec_format(modelFile, binary=True) #, binary=False
    
    sentList=[]
    sentList.append("disease affected people")
    sentList.append("Reports of affected people due to the disease")
    sentList.append("disease prevention")
    sentList.append("Questions or suggestions related to the prevention of disease or mention of a new prevention strategy")
    graph_sim(sentList,model,"test.graph.txt")
    
    
    
    