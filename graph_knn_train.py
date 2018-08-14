# -*- coding: utf-8 -*-
"""
@author: Firoj Alam
Modified: 12 Oct, 2016

"""



import sys
import datetime
from six import iteritems
import os
from os.path import basename
import numpy as np
import warnings
import logging
import operator
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print(__doc__)


    
def read_tweet_data_TSV(fileName):
    """
    Reads data from TSV file, then concates them by class-wise
    """
    data=[]
    with open(fileName, 'rU') as f:
        next(f)
        for line in f:
            line = line.strip()
            if (line==""):
                continue                
            row=line.split(",")
            data.append(row[:-1])
    return data
    

def create_NN_model(data, modeFileName, outFileName):
    model = NearestNeighbors(n_neighbors=100, algorithm='ball_tree',leaf_size=20,n_jobs=-1,metric='euclidean').fit(data)
    distances, indices = model.kneighbors(data)    
    #joblib.dump(model, modeFileName)
    textFile=open(outFileName,"w")        
    print("Writing into the file....")
    for i,vec in enumerate(indices):
        #instDict[vec[0]]=vec[1:]
        st=str(i)+" "
        if(i!=vec[0]):
            st=st+str(vec[0])+" "
        for val in vec[1:]:
            st=st+str(val)+" "
        textFile.write(st.strip()+"\n")
    textFile.close
    return outFileName
    
    

def load_model(filename):
    return joblib.load(filename)
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    a = datetime.datetime.now().replace(microsecond=0)
    args=sys.argv[1:-1]
    data=[]
    for fileName in args:
    	tmp=read_tweet_data_TSV(fileName)
    	data.extend(tmp)
    	
    #trFileName = sys.argv[1]; ## training file  
    outFile = sys.argv[-1]; # #out file
    
    #data=read_tweet_data_TSV(trFileName)
    print "Length: "+str(len(data))
    #data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    data = np.array(data,dtype=float)
    length=len(data)
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree',leaf_size=20,n_jobs=-1,metric='euclidean').fit(data)
    distances, indices = nbrs.kneighbors(data)
    #nbrs.kneighbors_graph(X).toarray()
    textFile=open(outFile,"w")        
    print("Writing into the file....")
    for i,vec in enumerate(indices):
        #instDict[vec[0]]=vec[1:]
        st=str(i)+" "
        if(i!=vec[0]):
            st=st+str(vec[0])+" "
        for val in vec[1:]:
            st=st+str(val)+" "
        textFile.write(st.strip()+"\n")
    textFile.close

    b = datetime.datetime.now().replace(microsecond=0)
    print "time taken:"
    print(b-a)    
    