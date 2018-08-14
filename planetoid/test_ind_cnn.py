import datetime
import time
from scipy import sparse as sp
import sys
sys.path.append("bin/cnn")
sys.path.append("bin/planetoid")
from ind_model_cnn import ind_model as model
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
import data_process as data_process
seed = 13
np.random.seed(seed)

#DATASET = 'citeseer'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 1.0)
parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 64)
parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 20)
parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 20)
parser.add_argument('--neg_samp', help = 'negative sampling rate; zero means using softmax', type = int, default = 0)
parser.add_argument('--g_learning_rate', help = 'learning rate for unsupervised loss', type = float, default = 1.0)
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

    
def getData(dataFile,delim):
    """
    Prepare the data
    """  
    train = pd.read_csv(dataFile, header=1, delimiter=delim)
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

def getAllData(dataFile,delim):
    """
    Prepare the data
    """  
    train = pd.read_csv(dataFile, header=1, delimiter=delim)
    R,C=train.shape
    x=train.iloc[:, 1:(C-1)]
    x=np.array(x.values)
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
    
start_time = time.time()
    
trainFile=args.train
devFile=args.dev    
testFile=args.test    
graphFile=args.graph
allxFile=args.allx
modeldir=args.modeldir
modeldirpath="models/"+modeldir+"/"
modeldirpath=checkdir(modeldirpath)


base=os.path.basename(trainFile)
tr_file_name=os.path.splitext(base)[0]

evaluation=checkdir("results_baseline/")
resultsFile=open(evaluation+tr_file_name+"_result.txt",'w')


MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
delim=","
data,_,_=data_process.getData(allxFile,delim)
word_index,tokenizer=data_process.getTokenizer(data,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
train_x,train_y,train_le,train_labels,_=data_process.getDevData2(trainFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
#delim="\t"
dev_x,dev_y,dev_le,dev_labels,_=data_process.getDevData2(devFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
test_x,test_y,test_le,test_labels,_=data_process.getDevData2(testFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
delim=","
allx,_,_,_,_=data_process.getDevData2(allxFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
graph=getGraph(graphFile)


print("Train: "+str(train_x.shape))
print("Dev: "+str(dev_x.shape))    
print("Test: "+str(test_x.shape))    
print("Graph: "+str(len(graph)))   
print("Allx: "+str(allx.shape))


m = model(args)                                                 # initialize the model
m.add_data(train_x, train_y, allx, graph,word_index) # add data
m.build()                                                       # build the model
m.init_train(init_iter_label = 10000, init_iter_graph = 200)    # pre-training
iter_cnt, max_accu = 0, 0
num_epochs=1000

patience = 150
best_valid = 0
best_valid_epoch = 0
best_weights = None
train_history={}
train_history['f1']=list()
train_history['epoch']=list()


epoch=0    
#while True:
for epoch in range(num_epochs):
    #m.step_train(max_iter = 1, iter_graph = 10, iter_inst = 100, iter_label = 10) # perform a training step
    m.step_train_minibatch(iter_label = 10)
    tpy = m.predict(dev_x)              # predict the dev set
    accu = comp_accu(tpy, dev_y)                                                   # compute the accuracy on the dev set
    print epoch, accu
    accu,P,R,F1,wAUC,AUC,report =performance.performance_measure_tf(dev_y,tpy,dev_le,dev_labels)
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    current_valid = train_history['f1'].append(f1_score)
    current_epoch = train_history['epoch'].append(epoch)
       
    if (epoch%10 == 0):
        current_valid = train_history['f1'][-1]
        print("f1-cur "+str(current_valid))
        current_epoch = train_history['epoch'][-1]
        if current_valid > best_valid:
            print("f1-best "+str(best_valid))
            best_valid = current_valid
            best_valid_epoch = current_epoch
            model=modeldirpath+tr_file_name+"_"+str(best_valid_epoch)+".model"
            #remove_prev_model("models/")
            print(model)
            m.store_params(model)
            best_weights = model
            resultsFile.write(model+"\n")
            result=str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(wauc))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n"
            resultsFile.write(result+"\n")
            resultsFile.write(report+"\n")
            print(result)
            print(report)
        elif best_valid_epoch + patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {} model {}.".format(best_valid, best_valid_epoch,best_weights))
            break        
print "**************Results on test set :: ST ::**************"        
m.load_params(best_weights)
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
seconds=time.time() - start_time
totol_time=str(datetime.timedelta(seconds=seconds))
resultsFile.write(totol_time+"\n")

#print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))

print(result)
print(result)
print "**************Results on test set :: END ::**************"
