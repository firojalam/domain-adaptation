#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:25:26 2017

@author: firojalam
"""
import numpy as np
from sklearn import metrics
import sys
import os
import sklearn.metrics as metrics
from sklearn import preprocessing
import pandas as pd
import re

def performance_measure(model,x_tst,y_tst,le,labels):   
    numClass=len(labels)
    pred_prob = np.empty((x_tst.shape[0], numClass))
    for ii in range(0, x_tst.shape[0]):
    	tvv = np.array(model.predict([x_tst[ii]]))
    	pred_prob[ii, ] = tvv[0]
    
    y_true=np.argmax(y_tst, axis = 1)
    y_pred=np.argmax(pred_prob, axis = 1)
    
    
    y_true=le.inverse_transform(y_true)
    lab=list(set(y_true.tolist()))
    lab.sort()    
    
    y_pred=le.inverse_transform(y_pred)
    acc=P=R=F1=0.0
    report=""
#    print y_true
    #print y_pred    
    AUC=metrics.roc_auc_score(y_true,y_pred, average="weighted")    
    try:
#        acc=metrics.accuracy_score(y_true,y_pred)   
#        P=metrics.precision_score(y_true,y_pred,average="weighted")        
#        R=metrics.recall_score(y_true,y_pred,average="weighted")    
#        F1=metrics.f1_score(y_true,y_pred,average="weighted")
        
        report=metrics.classification_report(y_true, y_pred)
        P,R,F1=classifaction_report(report)
    except Exception as e:
        print (e)
        pass        
    
    return acc,P,R,F1,AUC,report
    
def performance_measure_tf(y_true,tpy,le,labels):   
    #numClass=len(labels)    
    y_true=np.argmax(y_true, axis = 1)
    y_pred=np.argmax(tpy, axis = 1)        
    #lab=list(set(y_true.tolist()))
    #lab.sort()     
    wAUC,AUC=(0,0)
    y_pred_score=[]
    for scoreV,index in zip(tpy,y_pred):
        score=scoreV[index]
        y_pred_score.append(score)        
#    wAUC=metrics.roc_auc_score(y_true,y_pred_score,average='weighted') 
    wAUC=metrics.roc_auc_score(y_true,y_pred)     
    AUC=metrics.roc_auc_score(y_true,y_pred) 
    
    y_pred=le.inverse_transform(y_pred)
    y_true=le.inverse_transform(y_true)
    
#    print(AUC)
    
    acc=P=R=F1=0.0
    report=""
    try:
    	acc=metrics.accuracy_score(y_true,y_pred)   
	report=metrics.classification_report(y_true, y_pred)
    	P,R,F1=classifaction_report(report)
    except Exception as e:
        print (e)
        pass            
    
#    base=os.path.basename(outfile)
#    fname=os.path.splitext(base)[0]
#    pred_file="labeled/"+fname+"_pred.txt"
#    fopen = open(pred_file, "w");
#    fopen.write("##Ref.\tPrediction\n")        
#    for ref,pred in zip(y_true,y_pred):
#        fopen.write(str(ref)+"\t"+str(pred)+"\t1.0\n")
#    fopen.close    
    
    return acc,P,R,F1,wAUC,AUC,report 
    
    
def classifaction_report(report):
    report_data = []
    lines = report.split('\n')
#    print lines

    for line in lines[2:-3]:
        #print line
        line=line.strip()
        row = {}
        row_data = re.split('\s+', line)
#        print row_data
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    (P,R,F1,sumClassCnt)=(0,0,0,0)
    
    for row in report_data:
        tmp=row['precision']
        P=P+(tmp*row['support'])
        tmp=row['recall']
        R=R+(tmp*row['support'])
        tmp=row['f1_score']
        F1=F1+(tmp*row['support'])
        sumClassCnt=sumClassCnt+row['support']
    precision=P/sumClassCnt;
    recall=R/sumClassCnt;    
    f1_score=F1/sumClassCnt;        
#    print(str(precision)+"\t"+str(recall)+"\t"+str(f1_score)+"\n")
    
    #dataframe = pd.DataFrame.from_dict(report_data)
    #dataframe.to_csv('classification_report.csv', index = False)
    return precision, recall, f1_score
#report = classification_report(y_true, y_pred)
#classifaction_report_csv(report)