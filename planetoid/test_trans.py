
from scipy import sparse as sp
from trans_model import trans_model as model
import argparse
import cPickle
import optparse
import datetime
import pandas as pd
import numpy as np
import smart_open
import sklearn.metrics as metrics
from sklearn import preprocessing
#DATASET = 'citeseer'
import os
x=os.environ['LD_LIBRARY_PATH']
os.environ['LD_LIBRARY_PATH'] =x+":/usr/anaconda3/lib:/root/torch/install/lib:/usr/local/cuda/targets/x86_64-linux/lib"


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.1)
parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 64)
parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 64)
parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 30)
parser.add_argument('--neg_samp', help = 'negative sampling rate; zero means using softmax', type = int, default = 0)
parser.add_argument('--g_learning_rate', help = 'learning rate for unsupervised loss', type = float, default = 1e-2)
parser.add_argument('--model_file', help = 'filename for saving models', type = str, default = 'trans.model')
parser.add_argument('--use_feature', help = 'whether use input features', type = bool, default = True)
parser.add_argument('--update_emb', help = 'whether update embedding when optimizing supervised loss', type = bool, default = True)
parser.add_argument('--layer_loss', help = 'whether incur loss on hidden layers', type = bool, default = True)
parser.add_argument('--train', help = 'input train file', type = str, default = "")
parser.add_argument('--dev', help = 'input dev file', type = str, default = "")
parser.add_argument('--test', help = 'input test file', type = str, default = "")
parser.add_argument('--graph', help = 'input graph file', type = str, default = "")
args = parser.parse_args()

def comp_accu(tpy, ty):    
    return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]

def performance_measure(tpy, ty,le,tlabels):    
    y_true=np.argmax(ty, axis = 1)
    y_true=le.inverse_transform(y_true)
    lab=list(set(y_true.tolist()))
    lab.sort()    
    y_pred=np.argmax(tpy, axis = 1)
    y_pred=le.inverse_transform(y_pred)
    y_pred_score=[]
#    for scoreV,index in zip(tpy,y_pred):
#        score=scoreV[index]
#        y_pred_score.append(score)
    acc=P=R=F1=AUC=0.0
    report=""
    try:
        acc=metrics.accuracy_score(y_true,y_pred)   
        P=metrics.precision_score(y_true,y_pred,labels=lab,average="micro")        
        R=metrics.recall_score(y_true,y_pred,labels=lab,average="micro")    
        F1=metrics.f1_score(y_true,y_pred,labels=lab,average="micro")
        report=metrics.classification_report(y_true, y_pred)
    #print(metrics.classification_report(y_true, y_pred))
        #AUC=metrics.roc_auc_score(y_true,y_pred_score,labels=lab)    
    except Exception as e:
        print e
        pass        
    return acc,P,R,F1,AUC,report
    
#a = datetime.datetime.now().replace(microsecond=0)
#inparser = optparse.OptionParser()
#inparser.add_option('-i', action="store", dest="inFile")
#inparser.add_option('-o', action="store", dest="out_file")
#inparser.add_option('-m', action="store", dest="model")    
#options, args = inparser.parse_args()

def getData(dataFile):
    """
    Prepare the data
    """  
    train = pd.read_csv(dataFile, header=1, delimiter="," )  
    R,C=train.shape
    x=train.iloc[:, 0:(C-2)]
    x=np.array(x.values,dtype=np.float32)
    yL=train.iloc[:, C-1]
    
    le = preprocessing.LabelEncoder()
    yL=le.fit_transform(yL)
    labels=list(le.classes_)
    
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    return x,y,le,labels
    
def getGraph(dataFile):
    """
        Prepare the data
        """
    graphDict=dict()
    with smart_open.smart_open(dataFile) as f:
        count=0
        #next(f)
        for line in f:
            #if(count>100000):
            #    return graphDict
            line=line.strip()
            arr=line.split()
            #val=arr[0]
            val=int(arr[0])
            if(len(arr)==1):
                graphDict[val]=[]
            else:
                results = map(int, arr[1:10])
                graphDict[val]=results
            count=count+1
    return graphDict
            
def evaluate(m,tx,ty,le,tlabels):
    m.load_params()
    tpy = m.predict(tx)
    accu,P,R,F1,AUC,report=performance_measure(tpy,ty,le,tlabels)
    print accu, P, R, F1
    print report

trainFile=args.train
devFile=args.dev
graphFile=args.graph

x,y,le,labels=getData(trainFile)
tx,ty,le,tlabels=getData(devFile)
graph=getGraph(graphFile)

print("X length: "+str(x.shape))
print("Y length: "+str(y.shape))
print("graph length: "+str(len(graph)))


m = model(args)                                             # initialize the model
m.add_data(x, y, graph)                                     # add data
m.build()                                                   # build the model
m.init_train(init_iter_label = 2000, init_iter_graph = 2000)  # pre-training
iter_cnt, max_accu = 0, 0
while True:
    m.step_train(max_iter = 1, iter_graph = 0, iter_inst = 1, iter_label = 0)   # perform a training step
    tpy = m.predict(tx)                                                         # predict the dev set
    accu,P,R,F1,AUC,report=performance_measure(tpy, ty,le,tlabels)
    print iter_cnt, accu, max_accu, P, R, F1
    iter_cnt += 1
    if accu > max_accu:
        m.store_params()                                                        # store the model if better result is obtained
        max_accu = max(max_accu, accu)
        print iter_cnt, accu, max_accu, P, R, F1
        print report

print "results on test set"        
#if(args.test !=""):
testFile=args.test
tx,ty,le,tlabels=getData(testFile)
tpy = m.predict(tx)                                                         # predict the dev set
accu,P,R,F1,AUC,report=performance_measure(tpy, ty,le,tlabels)    
print iter_cnt, accu, max_accu, P, R, F1    

