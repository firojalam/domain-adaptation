
from scipy import sparse as sp
from ind_model import ind_model as model
import argparse
import cPickle
import optparse
import datetime
import pandas as pd
import numpy as np
import smart_open
import sklearn.metrics as metrics
from sklearn import preprocessing
import performance
import glob
import os
#DATASET = 'citeseer'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.01)
parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 100)
parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 100)
parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 100)
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
parser.add_argument('--allx', help = 'input feat file', type = str, default = "")
parser.add_argument('--modeldir', help = 'model file', type = str, default = "")
args = parser.parse_args()

def comp_accu(tpy, ty):    
    return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]

def performance_measure(tpy, ty):    
    y_true=np.argmax(ty, axis = 1)
    lab=list(set(y_true.tolist()))
    lab.sort()
    
    y_pred=np.argmax(tpy, axis = 1)
    y_pred_score=[]
    for scoreV,index in zip(tpy,y_pred):
        score=scoreV[index]
        y_pred_score.append(score)
    acc=P=R=F1=AUC=0.0
    try:
        acc=metrics.accuracy_score(y_true,y_pred)   
        P=metrics.precision_score(y_true,y_pred,labels=lab,average="micro")        
        R=metrics.recall_score(y_true,y_pred,labels=lab,average="micro")    
        F1=metrics.f1_score(y_true,y_pred,labels=lab,average="micro")
        AUC=metrics.roc_auc_score(y_true,y_pred_score,labels=lab)    
    except Exception as e:
        print e
        pass        
    return acc,P,R,F1,AUC
    
    
    

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
    print(x.shape)
    return x,y,le,labels

def getAllData(dataFile):
    """
    Prepare the data
    """  
    train = pd.read_csv(dataFile, header=1, delimiter="," )  
    R,C=train.shape
    x=train.iloc[:, 0:(C-1)]
    #print(x.shape)
#    yL=train.iloc[:, C-1]
#    label=yL.tolist()
    x=np.array(x.values,dtype=np.float32)
    return x
 
def get_range(dictionary, begin, end):
    data=dict()
    for k, v in dictionary.iteritems():
        if begin <= k <= end:
            arr=[]
            for item in v:
                if(item>end):
                    continue
                else:
                    arr.append(item)
            if(len(arr)==0):
                arr.append(k)
            data[k]=arr
    return data ##dict((k, v) for k, v in dictionary.iteritems() if begin <= k < end)
    
def getGraph(dataFile):
    """
    Prepare the data
    """  
    graphDict=dict()
    with smart_open.smart_open(dataFile) as f:
        for line in f:            
            line=line.strip()
            arr=line.split()
            #val=arr[0]            
            val=int(arr[0])
            if(len(arr)==1):
                graphDict[val]=[]
            else:
                results = map(int, arr[1:100])
                graphDict[val]=results
    return graphDict

def remove_prev_model(filename):
    files = glob.glob(filename+"/*")
    for f in files:
        os.remove(f)            
                
def checkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname  
    
trainFile=args.train
devFile=args.dev    
testFile=args.test    
graphFile=args.graph
allxFile=args.allx
modeldir=args.modeldir+"_feat"
modeldirpath="models/"+modeldir+"/"
modeldirpath=checkdir(modeldirpath)

base=os.path.basename(trainFile)
tr_file_name=os.path.splitext(base)[0]

evaluation=checkdir("results_baseline/")
resultsFile=open(evaluation+tr_file_name+"_result.txt",'w')

    
train_x,train_y,train_le,train_labels=getData(trainFile)
dev_x,dev_y,dev_le,dev_labels=getData(devFile)

graph=getGraph(graphFile)
allx=getAllData(allxFile)

#graph=get_range(graph,0,50000)
#allx=allx[0:50000,]
print(len(graph))
print(allx.shape)

m = model(args)                                                 # initialize the model
m.add_data(train_x, train_y, allx, graph)                                   # add data
m.build()                                                       # build the model
m.init_train(init_iter_label = 10000, init_iter_graph = 200)    # pre-training
iter_cnt, max_accu = 0, 0
num_epochs=2000

patience = 100
best_valid = 0
best_valid_epoch = 0
best_weights = None
train_history={}
train_history['f1']=list()
train_history['epoch']=list()

base=os.path.basename(trainFile)
tr_file_name=os.path.splitext(base)[0]

evaluation=checkdir("results_baseline/")
#results_file=open(evaluation+tr_file_name+"_result.txt",'w')

#for epoch in range(num_epochs):   
epoch=0    
#while True:
#for epoch in range(num_epochs):   
for epoch in range(num_epochs):     
    #m.step_train(max_iter = 100, iter_graph = 10, iter_inst = 10, iter_label = 10) # perform a training step
    m.step_train_minibatch(iter_label = 10)
    tpy = m.predict(dev_x)              # predict the dev set
    accu = comp_accu(tpy, dev_y)                                                   # compute the accuracy on the dev set
    iter_cnt += 1    
    accu,P,R,F1,wAUC,AUC,report = performance.performance_measure_tf(dev_y,tpy,dev_le,dev_labels)
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    current_valid = train_history['f1'].append(f1_score)
    current_epoch = train_history['epoch'].append(epoch)
    print (epoch)
    if (epoch%10 == 0):
        current_valid = train_history['f1'][-1]
        print("f1-cur "+str(current_valid))
        current_epoch = train_history['epoch'][-1]
        if current_valid > best_valid:
            print("f1-best "+str(best_valid))
            best_valid = current_valid
            best_valid_epoch = current_epoch
            model=modeldirpath+str(best_valid_epoch)+".model"
            print(model)
            m.store_params(model)
            best_weights = model
            resultsFile.write(model+"\n")
            result=str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(wauc))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n"
            resultsFile.write(result)
            resultsFile.write(report)
            print(result)
            print(report)
        elif best_valid_epoch + patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {} model {}.".format(best_valid, best_valid_epoch,best_weights))
            break
#    epoch=epoch+1  
    
    
print "**************Results on test set :: ST ::**************"          
m.load_params(best_weights)
testFile=args.test
test_x,test_y,test_le,test_labels=getData(testFile)
tpy = m.predict(test_x)                                                         # predict the dev set
accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(test_y,tpy,test_le,test_labels)    
wauc=wAUC*100
auc=AUC*100
precision=P*100
recall=R*100
f1_score=F1*100
resultsFile.write(testFile+"\n")
result=str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(wauc))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n"
resultsFile.write(result)
resultsFile.write(report)

print(result)
print(result)
print "**************Results on test set :: END ::**************"
