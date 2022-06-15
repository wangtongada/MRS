from __future__ import division
import pickle
import pandas as pd 
from math import isnan
import numpy as np
from math import lgamma
import itertools
from numpy.random import random
from random import sample
import time
from copy import deepcopy
import operator
from collections import Counter, defaultdict
from fim import fpgrowth,fim 
from scipy.sparse import csc_matrix

def predict_MRS(MRS,df):
    Z = np.zeros(len(df))
    for i,subdict in enumerate(MRS):
        tmp = np.zeros(len(df))
        for att in subdict.keys():
            tmp += (np.sum(df.iloc[:,subdict[att]],axis = 1)>0).astype(int)
        Z += (tmp == len(subdict)).astype(int)
    Yhat = (Z>0).astype(int)
    return Yhat

def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y)-FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP,FP,TN,FN

def predict_MRS_mix(MRS,df,itemNames,allthresholds,numericals):
    Z = np.zeros(len(df))
    for i,subdict in enumerate(MRS):
        tmp = np.zeros(len(df))
        for att in subdict.keys():
            if att in numericals:
                quantiles = [itemNames[item].split('_')[1] for item in subdict[att]]
                tmp += (np.sum([(np.array(df[att]>=allthresholds[att][int(q)])*np.array(df[att]<allthresholds[att][int(q)+1])).tolist() for q in quantiles],axis = 0)>0).astype(int)
            else:
                tmp += (np.sum(df[[itemNames[col] for col in subdict[att]]],axis = 1)>0).astype(int)
        Z += (tmp == len(subdict)).astype(int)
    Yhat = (Z>0).astype(int)
    return Yhat

def ROChull(points):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    fpr = []
    tpr = []
    index = np.unique(hull.simplices)
    for i in index:
        if points[i][0]<=points[i][1]:
            fpr.append(points[i][0])
            tpr.append(points[i][1])
    fpr,tpr = zip(*sorted(zip(fpr,tpr)))
    return fpr,tpr

def binary_code(df,collist,Nlevel):
    for q in range(1,Nlevel+1,1):
        thresholds = df[collist].quantile(float(q)/Nlevel)
        for col in collist:
            df[col+'_geq'+str(thresholds[col])] = (df[col] >= thresholds[col]).astype(float)
    df.drop(collist,axis = 1, inplace = True)

def code_intervals_nlevel(df,collist,Nlevel):
    for col in collist:
        for q in range(Nlevel):
            l = df[col].quantile(float(q)/Nlevel)
            u = df[col].quantile(float(q+1)/Nlevel)
            df[col+'_'+str(q)] = ((df[col] >=l) & (df[col]<u)).astype(int).values
    df.drop(collist,axis = 1, inplace = True)

def code_intervals_thresholds(df,collist,thresholds):
    for col in collist:
        for q in range(len(thresholds[col])-1):
            df[col+'_'+str(q)] = ((df[col] >=thresholds[col][int(q)]) & (df[col]<thresholds[col][int(q+1)])).astype(int).values
    df.drop(collist,axis = 1, inplace = True)

def log_dirmult(k,alpha):
    return  lgamma(sum(alpha)) - sum([lgamma(x) for x in alpha]) +sum([lgamma(alpha[i]+k[i]) for i in range(len(k))]) - lgamma(sum(k)+sum(alpha))

# alpha is the shape parameter and beta is the rate parameter
def log_gampoiss(k,alpha,beta):
    k = int(k)
    return lgamma(k+alpha)-lgamma(k+1)-lgamma(alpha)+alpha*(np.log(beta)-np.log(beta+1))-k*np.log(1+beta)
    # return lgamma(k+alpha)-lgamma(k+1)-k*np.log(1+beta)

def code_categorical(df,colnames,missingvalue):
    for col in colnames:
        values = np.unique(df[col])
        for val in values:
            if val not in missingvalue:
                df[col+'_'+str(val)] = (df[col]==val).astype(int)
    df.drop(colnames, axis = 1, inplace = True)

    def extract_rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]     

    def recurse(left, right, child, lineage=None):          
        if lineage is None:
            lineage = []
        if child in left:
            parent = np.where(left == child)[0].item()
            suffix = '_neg'
        else:
            parent = np.where(right == child)[0].item()
            suffix = ''

        #           lineage.append((parent, split, threshold[parent], features[parent]))
        lineage.append((features[parent].strip()+suffix))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)   
    rules = []
    for child in idx:
        rule = []
        for node in recurse(left, right, child):
            rule.append(node)
        rules.append(rule)
    return rules
