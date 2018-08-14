import sys
sys.path.append("bin/cnn")
sys.path.append("bin/planetoid")

from scipy import sparse as sp
from ind_model_domain_adaptation import ind_model as model
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
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer, get_output
from gensim.models import KeyedVectors
import lasagne
import theano.tensor as T
import theano

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 1.0)
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
parser.add_argument('--domain_data', help = 'domain data feat file', type = str, default = "")
parser.add_argument('--domain_test', help = 'domain test file', type = str, default = "")

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


def build_convpool_max(l_in,emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH):
    """
    Builds the complete network with maxpooling layer in time.
    :return: a pointer to the output of last layer
    """
    
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed    
    embedding_matrix=data_process.prepareEmbedding(word_index,emb_model,MAX_NB_WORDS,EMBEDDING_DIM)    
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)  
    print ("Number of words: "+str(nb_words))
    embedding_layer = lasagne.layers.EmbeddingLayer(l_in, 
                                                    input_size = nb_words, 
                                                    output_size = EMBEDDING_DIM,
                                                    W=embedding_matrix) #embedding_matrix) #lasagne.init.Normal()
    embedding_layer.params[embedding_layer.W].remove('trainable')
    #embedding_layer = DimshuffleLayer(embedding_layer, (0, 2, 1))
    #output = get_output(embedding_layer, x)
    print (embedding_layer.output_shape)
    convnets = [] # models to be merged
    filter_window_sizes=[2,3,4]
    num_filters=[100,150,200]
    for filter_len,nb_filter in zip(filter_window_sizes,num_filters):
        conv = Conv1DLayer(embedding_layer,nb_filter, filter_len, stride=1, pad='valid',W=lasagne.init.GlorotUniform(),nonlinearity=lasagne.nonlinearities.rectify)
        conv = lasagne.layers.MaxPool1DLayer(conv,pool_size=filter_len)
        conv = lasagne.layers.FlattenLayer(conv)        
        dense = lasagne.layers.DenseLayer(conv, nb_filter, W=lasagne.init.GlorotUniform(),nonlinearity = lasagne.nonlinearities.rectify)
        convnets.append(dense)
    print ("Conv done")
    convpool = lasagne.layers.ConcatLayer(convnets, axis = 1)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
#    convpool = lasagne.layers.dropout(convpool, p=0.02)
#    convpool = DenseLayer(convpool,num_units=512, nonlinearity=lasagne.nonlinearities.rectify)     
    print (convpool.output_shape)
    return convpool   
    
def iterate_minibatches(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]    

def predictions(ture,pred,le,outfile,ids):
    y_true=np.argmax(ture, axis = 1)
    y_pred=np.argmax(pred, axis = 1)        
    y_pred=le.inverse_transform(y_pred)
    y_true=le.inverse_transform(y_true)
    
    base=os.path.basename(outfile)
    fname=os.path.splitext(base)[0]
    pred_file="labeled/"+fname+"_pred.txt"
    fopen = open(pred_file, "w");
    fopen.write("##Ref.\tPrediction\n")        
    for id1,ref,pred in zip(ids,y_true,y_pred):
        fopen.write(str(id1)+"\t"+str(ref)+"\t"+str(pred)+"\t1.0\n")
    fopen.close       

def writeDataFile(Y,data,fileame):    
    #dirname=os.path.dirname(fileame)
    base=os.path.basename(fileame)
    name=os.path.splitext(base)[0]
    #out_file=dirname+"/"+name+"_"+featTxt+".csv"
    outFile="tsne/"+name+"_graph_dom_adapt.csv"
    print(outFile)
    featCol=[]
    feat=np.arange(data.shape[1])
    for f in feat:
        featCol.append("F"+str(f))
    index=np.arange(data.shape[0])
    
    df1 = pd.DataFrame(data,index=index,columns=featCol)
    df2 = pd.DataFrame(Y,index=index,columns=["class"])
    output=pd.concat([df1, df2], axis=1)
    output.to_csv(outFile, index=False, quoting=3) 
    return outFile
        
trainFile=args.train
devFile=args.dev    
testFile=args.test    
graphFile=args.graph
allxFile=args.allx
domain_data_file=args.domain_data
domain_test_file=args.domain_test


modeldir=args.modeldir+"_feat"
modeldirpath="models/"+modeldir+"/"
modeldirpath=checkdir(modeldirpath)


base=os.path.basename(trainFile)
tr_file_name=os.path.splitext(base)[0]

evaluation=checkdir("results_baseline/")
resultsFile=open(evaluation+tr_file_name+"_result.txt",'w')


MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
batch_size=128
delim="\t"
data,_,_=data_process.getData(allxFile,delim)
word_index,tokenizer=data_process.getTokenizer(data,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
train_x,train_y,train_le,train_labels,_=data_process.getDevData2(trainFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
delim="\t"
dev_x,dev_y,dev_le,dev_labels,_=data_process.getDevData2(devFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
test_x,test_y,test_le,test_labels,_=data_process.getDevData2(testFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
delim="\t"
allx,_,_,_,_=data_process.getDevData2(allxFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)

graph=getGraph(graphFile)

MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
#emb_file="/export/home/fialam/crisis_semi_supervised/crisis-tweets/model/crisis_word_vector.txt"  
modelFile="../w2v_models/crisis_word_vector.txt"   
emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)
print("Loaded embedding matrix")

x_sym = T.imatrix( 'inputs' )
l_in = lasagne.layers.InputLayer((None, MAX_SEQUENCE_LENGTH),x_sym)
model2=build_convpool_max(l_in,emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH)    
output = get_output(model2, x_sym)

cnn = theano.function([x_sym], output)
train_x=cnn(train_x)
dev_x=cnn(dev_x)
test_x=cnn(test_x)

#allx=cnn(allx)
allx_data=[]
for batch in iterate_minibatches(allx,batch_size):
    allx_data.extend(cnn(batch))
allx_data=np.array(allx_data,dtype = 'float32')    
print(len(graph))
print(allx_data.shape)

delim="\t"
domain_test_x,domain_test_y,domain_test_le,domain_test_labels,_=data_process.getDevData2(domain_test_file,tokenizer,MAX_SEQUENCE_LENGTH,delim)
domain_data_x,_,_,_,_=data_process.getDevData2(domain_data_file,tokenizer,MAX_SEQUENCE_LENGTH,delim)

domain_data=cnn(domain_data_x)
domain_test=cnn(domain_test_x)
    
print(len(graph))
print(allx.shape)

m = model(args)                                                 # initialize the model
m.add_data(train_x, train_y, allx_data, graph,domain_data)                                   # add data
m.build()                                                       # build the model
m.init_train(init_iter_label = 10000, init_iter_graph = 400)    # pre-training
iter_cnt, max_accu = 0, 0
num_epochs=1000

patience = 150
best_valid = 0
best_valid_epoch = 0
best_weights = None
train_history={}
train_history['f1']=list()
train_history['epoch']=list()

#for epoch in range(num_epochs):   
epoch=0    
#while True:
for epoch in range(num_epochs):    
    #m.step_train(max_iter = 100, iter_graph = 10, iter_inst = 10, iter_label = 10) # perform a training step
    m.step_train_minibatch(iter_label = 10,epoch=epoch)
    tpy = m.predict(dev_x)              # predict the dev set
    accu = comp_accu(tpy, dev_y)                                                   # compute the accuracy on the dev set
    iter_cnt += 1    
    accu,P,R,F1,wAUC,AUC,report =performance.performance_measure_tf(dev_y,tpy,dev_le,dev_labels)
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    current_valid = train_history['f1'].append(f1_score)
    current_epoch = train_history['epoch'].append(epoch)
#    print (epoch)
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
print "**************Results on test set :: END ::**************"

print "**************Results on domain test set :: ST ::**************"          
tpy = m.predict(domain_test)                                                         # predict the dev set
accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(domain_test_y,tpy,domain_test_le,domain_test_labels)    
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
print "**************Results on domain test set :: END ::**************"


data=[]    
Y=[]

train_x_last_layer = m.predict_last_layer(train_x)    
print (train_x_last_layer.shape)
source_tr_y=[0] * len(train_x_last_layer)    
domain_last_l = m.predict_last_layer(domain_test) 
domain_tr_y=[1] * len(domain_last_l)

data.extend(train_x_last_layer)
data.extend(domain_last_l)
Y.extend(source_tr_y)
Y.extend(domain_tr_y)
Y=np.array(Y,'int32')
data=np.array(data)
domain_tr_y=domain_last_l.astype('int32')

writeDataFile(Y,data,trainFile)


