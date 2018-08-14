#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 13:45:26 2017

@author: firojalam
"""


import numpy as np
seed = 1337
np.random.seed(seed)

def make_feature_vec(words, model, EMBEDDING_DIM):
    featureVec = np.zeros((EMBEDDING_DIM,),dtype="float32")
    for word in words:
        try:
            featureVec=np.c_[ featureVec, model[word] ]
        except Exception:
            try:
                rng = np.random.RandomState()        	
                embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)
                featureVec=np.c_[ featureVec, embedding_vector ]
            except KeyError:
                continue      
    mean = np.zeros((EMBEDDING_DIM),dtype="float32")
    if(not featureVec.any()):
        mean=np.random.rand(EMBEDDING_DIM,)            
    elif(featureVec.shape[0]>1):
        mean=np.mean(featureVec,axis=1)
    elif(featureVec.shape[0]==1):
        mean=featureVec
    return mean
    
def get_avg_feature_vecs(texts, model, EMBEDDING_DIM):
 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(texts),EMBEDDING_DIM),dtype="float32")
    counter=0;
    for review in texts:
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(texts))
       if(len(review)>0):
           vec=make_feature_vec(review, model, EMBEDDING_DIM)
           reviewFeatureVecs[counter]=vec         
           counter=counter+1
       else:
           reviewFeatureVecs[counter]=np.random.rand(EMBEDDING_DIM,)
           counter=counter+1
    return reviewFeatureVecs    