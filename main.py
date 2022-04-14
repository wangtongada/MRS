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
from util import *


class MRS(object):
    def __init__(self,data,Y):
        self.attributeLevelNum = defaultdict(int) 
        self.attributeLevelNames = defaultdict(list)
        self.attributeLevelIndeces = defaultdict(list)
        self.itemNames = defaultdict(list)
        self.colindices = defaultdict(list)
        self.attributeNames = []
        self.df = binary_data 
        self.Y = Y
        self.N = len(Y)
      
    def __init__(self, binary_data,Y):
        self.attributeLevelNum = defaultdict(int) 
        self.attributeLevelNames = defaultdict(list)
        self.attributeLevelIndeces = defaultdict(list)
        self.itemNames = defaultdict(list)
        self.colindices = defaultdict(list)
        self.attributeNames = []
        self.df = binary_data 
        self.Y = Y
        self.N = len(Y)

        for i,name in enumerate(self.df.columns):
            attribute = name.split('_')[0]
            self.attributeNames.append(attribute)
            value = ''.join(name.split('_')[1:])
            self.attributeLevelNum[attribute] += 1
            self.attributeLevelNames[attribute].append(value)
            self.attributeLevelIndeces[attribute].append(i)
            self.itemNames[i] = name
            self.colindices[name] = i
            # self.colindices[name+'neg']=2*i + 1
        self.attributeNames = list(set(self.attributeNames))  


# mean lambda = alpha/beta
    def set_parameters(self, alpha_1=100,beta_1=1,alpha_2=100,beta_2=1,alpha_M=1,beta_M=100,alpha_L=1,beta_L=2,alpha = []):
        self.alpha_1,self.beta_1= alpha_1,beta_1 
        self.alpha_2,self.beta_2= alpha_2,beta_2 
        self.beta_M,self.alpha_M = beta_M,alpha_M 
        self.beta_L,self.alpha_L = beta_L,alpha_L 
        # if alpha==None:
        self.alpha = [1 for i in range(len(self.attributeNames))]

    # def set_parameters(self, rho1 = 0.9, rho2 = 0.9,alpha_M=1,beta_M=100,alpha_L=1,beta_L=2,alpha = []):
    #     self.rho1,self.rho2 = rho1, rho2
    #     self.beta_M,self.alpha_M = beta_M,alpha_M 
    #     self.beta_L,self.alpha_L = beta_L,alpha_L 
    #     # if alpha==None:
    #     self.alpha = [1 for i in range(len(self.attributeNames))]

    def generate_rules(self,supp,maxlen,N, need_negcode = False,njobs = 5, method = 'fpgrowth',criteria = 'IG',add_rules = []):
        self.maxlen = maxlen
        self.supp = supp
        if method =='fpgrowth':
            print('Using fpgrowth to generate rules with support {} and max length {}'.format(supp,maxlen))
            itemMatrix = [[item for item in self.df.columns if row[item] ==1] for i,row in self.df.iterrows() ]  
            pindex = np.where(self.Y==1)[0]
            nindex = np.where(self.Y!=1)[0]
            start_time = time.time()
            rules= fpgrowth([itemMatrix[i] for i in pindex],supp = supp,zmin = 1,zmax = maxlen)
            rules = [np.sort(x[0]).tolist() for x in rules]
            df = self.df
        else:
            print('Using random forest to generate rules ...')
            rules = []
            start_time = time.time()

            for length in range(2,maxlen+1,1):
                n_estimators = 500*length# min(5000,int(min(comb(df.shape[1], length, exact=True),10000/maxlen)))
                clf = RandomForestClassifier(n_estimators = n_estimators,max_depth = length)
                clf.fit(self.df.iloc[:,list(range(int(self.df.shape[1]/2)))],self.Y)
                for n in range(n_estimators):
                    rules.extend(extract_rules(clf.estimators_[n],self.df.columns[:int(self.df.shape[1]/2)]))
            rules = [list(x) for x in set(tuple(np.sort(x)) for x in rules)]   
            df = 1-self.df 
            df.columns = [name.strip() + 'neg' for name in self.df.columns]
            df = pd.concat([self.df,df],axis = 1)

        self.generate_time = time.time() - start_time
        print('\tTook %0.3fs to generate %d rules' % (self.generate_time, len(rules)))
        count = 0
        index = []
        for rule in add_rules:
            if np.sort(rule).tolist()  not in rules:
                rules.append(rule)
                index.append(len(rules)-1)
            else:
                index.append(rules.index(rule))
        self.rulespace = [len(rules)]
        self.all_rulelen = np.array([len(rule) for rule in rules])
        self.screen_rules(rules,df,N,supp,criteria,
                          ,index) # select the top N rules using secondary criteria, information gain

    def screen_rules(self,rules,df,N,supp,criteria = 'precision',njobs = 5,add_rules = []):
        # print 'screening rules'
        start_time = time.time()
        itemInd = {}
        for i,name in enumerate(df.columns):
            itemInd[name] = int(i)
        len_rules = [len(rule) for rule in rules]
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule] for rule in rules])))
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(df.columns),len(rules)))
        mat = np.matrix(df) * ruleMatrix
        lenMatrix = np.matrix([len_rules for i in range(df.shape[0])])
        Z =  (mat ==lenMatrix).astype(int)

        Zpos = [Z[i] for i in np.where(self.Y>0)][0]
        TP = np.array(np.sum(Zpos,axis=0).tolist()[0])
        # supp_select = np.where(TP>=supp*sum(self.Y)/100.0)[0]
        supp_select = np.array([i for i in range(len(TP))])
        if len(supp_select)<=N:
            self.rules = [rules[i] for i in supp_select]
            self.RMatrix = np.array(Z[:,supp_select])
            self.rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in self.rules]
            self.supp = np.array(np.sum(Z,axis=0).tolist()[0])[supp_select]
        else:
            FP = np.array(np.sum(Z,axis = 0))[0] - TP
            TN = len(self.Y) - np.sum(self.Y) - FP
            FN = np.sum(self.Y) - TP
            p1 = TP.astype(float)/(TP+FP)
            p2 = FN.astype(float)/(FN+TN)
            pp = (TP+FP).astype(float)/(TP+FP+TN+FN)
            self.precision = p1

            if criteria =='precision':
                select = np.argsort(p1[supp_select])[::-1][:N].tolist()
            else: # 
                cond_entropy = -pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1))-(1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2))
                cond_entropy[p1*(1-p1)==0] = -((1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2)))[p1*(1-p1)==0]
                cond_entropy[p2*(1-p2)==0] = -(pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1)))[p2*(1-p2)==0]
                cond_entropy[p1*(1-p1)*p2*(1-p2)==0] = 0
                pos = (TP+FN).astype(float)/(TP+FP+TN+FN)
                info = - pos * np.log(pos) - (1-pos)*np.log(1-pos)
                info[np.where((pos==1)| (pos==0))[0]] = 0
                IGR = (info - cond_entropy)/info 
                IGR[np.where(info==0)[0]] = 0
                select = np.argsort(IGR[supp_select])[::-1][:N].tolist()
            ind = list(supp_select[select])
            self.rules = [rules[i] for i in ind]
            self.RMatrix = np.array(Z[:,ind])
            self.score = p1[select]
            self.TP = TP
            self.rules_len = [len(set([name.split('_')[0] for name in rule])) for rule in self.rules]
            self.supp = np.array(np.sum(Z,axis=0).tolist()[0])[ind]
        self.screen_time = time.time() - start_time
        print('\tTook %0.3fs to generate %d rules' % (self.screen_time, len(self.rules)))

    def precompute(self):
        # compute L0
        TP,FP,TN,FN = sum(self.Y),0,len(self.Y) - sum(self.Y),0
        self.Lup = log_betabin(TP,TP+FP,self.alpha_1,self.beta_1)+ log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
        # self.Lup = TP*np.log(self.rho1) + FP*np.log(1-self.rho1) + TN*np.log(self.rho2) + FN*np.log(1-self.rho2)
        # self.Lllh0 = sum(self.Y)*np.log(1-self.rho2)+ (len(self.Y) - sum(self.Y))*np.log(self.rho2) # fix it later
        self.LP0 =  np.log(pow(np.true_divide(self.beta_M,self.beta_M+1),self.alpha_M) ) 
        # self.LUpsilon = np.log(1-self.rho2) - np.log(self.rho1)       
        self.LUpsilon = np.log(np.true_divide((sum(self.Y)+self.alpha_1+self.beta_1+1)*self.beta_2,(sum(self.Y)+self.alpha_1-1)*(len(self.Y)-sum(self.Y)+self.alpha_2+self.beta_2)))
        self.Omega = np.true_divide((self.beta_M+1)*pow(self.beta_L+1, self.alpha_L+1)*sum(self.alpha),self.alpha_M*self.alpha_L*pow(self.beta_L,self.alpha_L)*max(self.alpha))
        # self.M = [np.ceil(np.true_divide(self.Lup - self.Lllh0,np.log(self.Omega)))] # fix it later
        self.M = [1000]
        self.minsupp = [1]

    # def order_subset(self):
    #     self.attribute_superset = defaultdict(list)
    #     for att in self.attributeLevelIndeces.keys():
    #         self.attribute_superset[att]=[]

        # self.attribute_superset = defaultdict(list)
        # for att in self.attributeLevelIndeces.keys():
        #     for col in self.attributeLevelIndeces[att]:
        #         self.attribute_superset[col] = []
        #         for sup in [x for x in self.attributeLevelIndeces[att] if x != col]:
        #             if sum(self.df.ix[:,sup]>=self.df.ix[:,col])==self.N and sum(self.df.ix[:,sup]==self.df.ix[:,col])<self.N:
        #                 self.attribute_superset[col].append(sup)

    def train(self, Niteration = 2000, q=0.3, print_message=False,init = []):
        self.precompute()
        start_time = time.time()
        self.Y = np.array(self.Y)
        """ parameters for Simulated Annealing """
        T0 = 10000
        pt_max = -100000000
        self.maps=[]
        if init == []:
            N = sample(range(1,8,1),1)[0]
            Lns = [sample(range(1,5,1),1)[0] for n in range(N)]
            MRS_curr=[]
            for n in range(N):
                attributes = sample(self.attributeNames, Lns[n])
                subdict = defaultdict(list)
                for attr in attributes:
                    subdict[attr].append(sample(self.attributeLevelIndeces[attr],1)[0])
                MRS_curr.append(subdict)
        else:
            MRS_curr = init
        self.MRS_curr = MRS_curr
        MRS_curr,_,prob,Yhat_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = self.compute_prob(MRS_curr)
        pt_curr = sum(prob)
        self.memory = {}
        for iter in range(Niteration):
            
            if print_message:
                print('========= iteration = {} ========='.format(iter))
            # MRS_new_len = len(MRS_curr)
            if iter>0.75*Niteration:
                Yhat_curr,pt_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = Yhat_max.copy(),pt_max,W_max.copy(),V_max.copy(),MRS_max_rlens.copy(),MRS_max_len
                MRS_curr = [subdict for subdict in MRS_max]
            MRS_new = [deepcopy(subdict) for subdict in MRS_curr]
            # self.MRS_new = [deepcopy(subdict) for subdict in MRS_curr]
            if print_message:
                print('before action, the new set is:')
                self.printMRS(MRS_new)
            added_rule = False
            self.memory[iter] = MRS_new
            if random()>np.exp(-np.true_divide(iter,1.5*Niteration)) or MRS_curr_len ==1: # proposing a new solution
                incorr = np.where(self.Y!=Yhat_curr)[0] # collect misclassified example
                if len(incorr)==0: # every data point is correctly classified
                    trim = False
                    break
                    # it means the MRS correctly classified all points but the rule could be redundant, so triming is needed
                else:
                    trim = False
                    ex = sample(list(incorr),1)[0] # sample from misclassified 
                    ex_dict = defaultdict(list) # conditions that ex satisfies
                    for i,name in enumerate(self.df.columns):
                        attr_name = name.split('_')[0]
                        if self.df.iloc[ex,i]==1:
                            ex_dict[attr_name].append(i)
                    if print_message:
                        print('ex={}'.format(ex))
                if random()<0 and MRS_curr_len>1:
                    # print('perturb')
                    # self.action = 'action_perturb'
                    self.action_perturb(MRS_new,W_curr)
                else:
                    if trim == False and random()<=0: # jump out of local maximum:
                        # self.action = 'action_jump'
                        self.action_jump(MRS_new,W_curr,q,print_message)
                    elif trim==False and self.Y[ex]==1: # it means the MRS failed to cover it, so we need to do one of the following
                         # find out which of the rules does ex satisfy all but one condition
                        t = random()
                        if t<.333: # add value to cover ex
                            # ind = sample(list(temp),1)[0]
                            # print('add value')
                            temp  = [i for i in range(MRS_curr_len) if V_curr[i][ex] == MRS_curr_rlens[i]-1]
                            # self.action = 'action_addvalue'
                            self.action_addvalue(MRS_new,ex_dict,Yhat_curr,temp,q,print_message)
                        elif t<.667 and not (MRS_curr_len==1 and len(MRS_new[0])==1) and MRS_curr_len>0: # action1:remove a different condition from a rule
                            # print('remove condition')
                            # self.action = 'action_rmcondition'
                            self.action_rmcondition(MRS_new,V_curr,W_curr,ex_dict,print_message) # problematic to do that, DO NOT REMOVE CONDITION
                        else: # add a rule
                            # print('add rule')
                            # self.action = 'action_addrule'
                            self.action_addrule(MRS_new,W_curr,Yhat_curr,ex,ex_dict,self.maxlen,q,print_message)
                            added_rule = True
                    else: # it means the MRS covers the wrong data, so we need to do one of the following 
                        t = random()
                        if trim==False and (MRS_curr_len<=1 or t<=0.333):#or (random()<=1.0/(2-float(iter)/Niteration) and MRS_curr_len<=10: # action3: add a condition to a rule that satisfies the example to make it more specific
                            # print('add condition')
                            # self.action = 'action_addcondition'
                            self.action_addcondition(MRS_new,ex_dict,W_curr,print_message)
                        elif t<= 0.667: # action4: remove a rule from MRS
                            # print('remove rule')
                            # self.action = 'action_rmrule'
                            self.action_rmrule(MRS_new,W_curr,print_message)
                        else: # replace a rule
                            # self.action = 'action_rmrule'
                            self.action_rmrule(MRS_new,W_curr,print_message)
                            # self.MRS_compute = [subdict for subdict in MRS_new]
                            MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new)
                            incorr = np.where(np.logical_and(self.Y==1, Yhat_new==0))[0]
                            ex = sample(list(incorr),1)[0]
                            ex_dict = defaultdict(list) # conditions that ex satisfies
                            for i,name in enumerate(self.df.columns):
                                attr_name = name.split('_')[0]
                                if self.df.iloc[ex,i]==1:
                                    ex_dict[attr_name].append(i)
                            # self.action = 'action_addrule'
                            # print('2nd add rule')
                            self.action_addrule(MRS_new,W_new,Yhat_new,ex,ex_dict,self.maxlen,q,print_message)
                            added_rule = True
                # self.MRS_compute = [subdict for subdict in MRS_new]
                MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new)
                # pt_new = sum(prob)
                # if pt_new < pt_curr and random()<0.5:
                #     self.refine_rule(MRS_new,W_new,print_message)
            else:
                # self.action = 'action_trymerge'
                # print('try merge')
                self.action_trymerge(MRS_new)
            MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new)
            if random()<0.25 and added_rule == True:
                self.refine_rule(MRS_new,W_new,print_message)
                MRS_new,cfmatrix,prob,Yhat_new,W_new,V_new,MRS_new_rlens,MRS_new_len= self.compute_prob(MRS_new)
            pt_new = sum(prob)
            T = np.true_divide(T0, 1 + iter)
            alpha = np.exp(float(pt_new -pt_curr)/T)
            if print_message:
                print('original MRS is')
                self.printMRS(MRS_curr)
                print('new MRS is:')
                self.printMRS(MRS_new)
                # print('merged MRS is:')
                # self.printMRS(MRS_merged)
                TP,FP,TN,FN = cfmatrix
                tpr = float(TP)/(TP+FN)
                fpr = float(FP)/(FP+TN)
                print('accuracy = {}, tpr = {}, fpr = {}, TP = {},FN = {}, FP = {}, TN = {}\n pt_new is {}, prior_NumOfRules= {}, prior_NumOfItems = {}, prior_ChsValues = {}, likelihood_1 = {}, likelihood_2 = {}\n'.format(float(TP+TN)/self.N,tpr,fpr,TP,FN,FP,TN,pt_new, prob[0], prob[1],prob[2],prob[3], prob[4]))

            if pt_new > pt_max:
                TP,FP,TN,FN = cfmatrix
                tpr = float(TP)/(TP+FN)
                fpr = float(FP)/(FP+TN)
                MRS_max = [deepcopy(subdict) for subdict in MRS_new]
                # MRS_max_merged = [copy.deepcopy(subdict) for subdict in MRS_merged]
                pt_max,Yhat_max, W_max,V_max, MRS_max_rlens, MRS_max_len = pt_new, Yhat_new.copy(),W_new.copy(),V_new.copy(),MRS_new_rlens.copy(),MRS_new_len

                self.MRS_max = MRS_max
                self.M.append(np.ceil(np.true_divide(self.Lup +self.LP0 - pt_new,np.log(self.Omega))))
                self.minsupp.append(np.ceil(np.true_divide(np.log(np.true_divide(self.M[-1]+self.alpha_M-1,self.M[-1]*self.alpha_M*self.Omega)),self.LUpsilon)))
                self.rulespace.append(len(np.where(self.all_rulelen>= self.minsupp[-1])[0]))
                print('\n** max at iter = {} ** \nTP = {},FN = {}, FP = {}, TN = {}\n pt_new is {}, prior_NumOfRules= {}, prior_LenOfRules = {}, prior_ChsItems = {}, likelihood_1 = {}, likelihood_2 = {}\n accuracy = {}, tpr = {}, fpr = {}\n M = {}, supp = {}'.format(iter,TP,FN,FP,TN,pt_new, prob[0], prob[1],prob[2],prob[3], prob[4],float(TP+TN)/(FP+TP+TN+FN),tpr,fpr,self.M[-1],self.minsupp[-1]))
                # print('Size = {}, supp = {}'.format(self.M[-1],self.supp[-1]))
                print('merged MRS_merged is')
                self.MRS_max = MRS_max
                self.printMRS(MRS_max)
                print([sum(W_max[i]) for i in range(len(W_max))])
                self.maps.append([iter,pt_new,MRS_new,cfmatrix, self.M[-1],self.minsupp[-1],time.time() - start_time])
            if random() <= alpha:
                # print 'rules_curr is {},flip={}, threshold = {} move is {}'.format(rules_curr,flip,threshold,move)
                MRS_curr = [deepcopy(subdict) for subdict in MRS_new]
                Yhat_curr,pt_curr,W_curr,V_curr,MRS_curr_rlens,MRS_curr_len = Yhat_new.copy(),pt_new,W_new.copy(),V_new.copy(),MRS_new_rlens.copy(),MRS_new_len
                if print_message:
                    print('ACCEPT the new MRS\n')
        MRS_merged = self.merge(MRS_max)
        return MRS_merged,self.maps

    def refine_rule(self,MRS_new,W,print_message):
        if print_message:
            print('Select the BEST condition to add')
        score_max = 0
        add_i = -1
        for i,subdict in enumerate(MRS_new):
            # InfoGain_i = compute_InfoGain(W[i],self.Y)
            if sum(W[i])>self.minsupp[-1]: # Contrapositive property
                candidate_attributes =[i for i in self.attributeNames if i not in subdict.keys()]
                for attr in candidate_attributes:
                    candidate_values = self.attributeLevelIndeces[attr]
                    for value in candidate_values:
                        # self.W = W
                        # self.i = i
                        TP,FP,TN,FN = getConfusion(W[i],self.Y)
                        score1 = float(TP)/(TP+FP+1)
                        Z = W[i]*self.df.iloc[:,value]
                        TP,FP,TN,FN = getConfusion(Z,self.Y)
                        score2 = float(TP)/(TP+FP+1) #log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
                        score_d = score2 - score1
                        if score_d>=score_max and TP+FP >self.minsupp[-1]:
                            score_max = score_d
                            add_attribute = attr
                            add_value = value
                            add_i = i
        if add_i>=0:
            if add_value not in MRS_new[add_i][add_attribute]:
                MRS_new[add_i][add_attribute].append(add_value)
            if print_message:
                print('refine rules - add condition: add value {} to attribute {} in rule i = {}, score = {}'.format(add_value,add_attribute,add_i,score_max))

    def action_addvalue(self,MRS_new,ex_dict,Yhat,indices,q,print_message):
        score_max = -1
        neg_ind = np.where(Yhat==0)[0]
        add_val = -1
        if print_message:
            print('\n add value ')
        for ind in indices:
            for att in MRS_new[ind].keys():
                if not any([x in ex_dict[att] for x in MRS_new[ind][att]]):
                    candidate_values = [x for x in ex_dict[att] if x not in MRS_new[ind][att]]
                    for val in candidate_values:
                        values = MRS_new[ind][att]+[val]
                        Z = (np.sum(self.df.iloc[:,values],axis = 1)>0).astype(int).values 
                        if sum(Z)<self.N:
                            TP,FP,TN,FN = getConfusion(Z[neg_ind],self.Y[neg_ind])
                            score_temp = float(TP)/(TP+FP+1)
                            # TP,FP,TN,FN = getConfusion(Yhat_temp[nindex],Y[nindex])
                            # score_temp = log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2)
                            # print 'score={},add_i={},attr={},value={}'.format(score_temp,add_i,attr,value)
                            if score_temp>=score_max:
                                score_max = score_temp
                                add_ind = ind
                                add_att = att
                                add_val = val
        if score_max>0 and add_val in MRS_new[add_ind][add_att]:
            score_max = -1
        if add_val<0 or score_max == -1:
            ind_candidates = [i for i in range(len(MRS_new)) if i not in indices]
            for ind in ind_candidates:
                for att in MRS_new[ind].keys():
                    val_candidates = [x for x in self.attributeLevelIndeces[att] if x not in MRS_new[ind][att]]
                    for val in val_candidates:
                        values = MRS_new[ind][att]+[val]
                        Z = (np.sum(self.df.iloc[:,values],axis = 1)>0).astype(int).values
                        if sum(Z)<self.N:
                            TP,FP,TN,FN = getConfusion(Z[neg_ind],self.Y[neg_ind])
                            score_temp = float(TP)/(TP+FP+1)
                            if score_temp>=score_max:
                                score_max = score_temp
                                add_ind = ind
                                add_att = att
                                add_val = val
        if add_val>=0:
            MRS_new[add_ind][add_att].append(add_val)
            if print_message:
                print('-- add value {} to feature \'{}\' in rule {} --'.format(add_val,add_att,add_ind))
        return                   

    def action_perturb(self,MRS_new,W):
        supp = np.array([sum(row) for row in W])
        ind = np.where(supp == min(supp))[0][0]
        MRS_new.pop(ind)

    def action_trymerge(self, MRS_new):
        self.MRS_new = MRS_new
        l = len(MRS_new)
        Z = np.array([[False]*len(self.attributeNames) for i in range(l)])
        for i,subdict in enumerate(MRS_new):
            Z[i][[int(att in MRS_new[i].keys()) for att in self.attributeNames]] = True
        dist = np.ones((l,l))*len(self.attributeNames) 
        values = []
        for i in range(l):
            for j in range(i+1,l,1):
                x = sum([x^y for x,y in zip(Z[i],Z[j])])
                dist[i,j] = x
                values.append(x)
        dist = dist/float(self.df.shape[1])
        d = 2*(1+max(values) - min(values))/float(self.df.shape[1])
        dist = dist + np.multiply(d,np.random.rand(l,l))
        r1,r2 = np.unravel_index(dist.argmin(), dist.shape)
        attributes = [x for x in MRS_new[r1].keys() if (x in MRS_new[r2].keys() and set(MRS_new[r2][x])!=set(MRS_new[r1][x]))]
        if len(attributes)>0:
            att = sample(attributes,1)[0]
            vals = list(set(MRS_new[r2][att]).union(set(MRS_new[r1][att])))
            MRS_new[r2][att] = vals.copy()
            MRS_new[r1][att] = vals.copy()
            return
        if len(Z[r1])<=len(Z[r2]): # r1 is the shorter rule
            attributes = MRS_new[r1].keys()
            if len(attributes)>0:
                while True:
                    att = sample(attributes,1)[0]
                    attributes = [x for x in attributes if x!= att]
                    if att not in MRS_new[r2].keys():
                        MRS_new[r2][att] = MRS_new[r1][att]
                        break
                    else:
                        vals = list(set(MRS_new[r2][att]).union(set(MRS_new[r1][att])))
                        MRS_new[r2][att] = vals.copy()
                        MRS_new[r1][att] = vals.copy()
                        break
        else:
            attributes = MRS_new[r2].keys()
            if len(attributes)>0:
                while True:
                    att = sample(attributes,1)[0]
                    attributes = [x for x in attributes if x!= att]
                    if att not in MRS_new[r1].keys() :
                        MRS_new[r1][att] = MRS_new[r2][att]
                        break
                    else:
                        vals = list(set(MRS_new[r1][att]).union(set(MRS_new[r2][att])))
                        MRS_new[r2][att] = vals.copy()
                        MRS_new[r1][att] = vals.copy()
                        break


    def action_jump(self,MRS_new,W,q,print_message):
        if random()<q: # randomly select a rule to add a condition
            if print_message:
                print('in jump: Select a RANDOM item to add to a random rule')
            add_i = sample(range(len(MRS_new)),1)[0]
            add_attribute = sample(set(self.attributeNames).difference(set(MRS_new[add_i].keys())),1)[0]
            add_value = sample(self.attributeLevelIndeces[add_attribute],1)[0]
        else:
            l = len(MRS_new)
            if print_message:
                print('in jump: Select the BEST condition to add')
            score_max = -10000
            add_i = -1
            p_candidate = []
            for i in range(l):
                TP,FP,TN,FN = getConfusion(W[i],self.Y)
                p_candidate.append(FP)
            p_candidate = np.insert(p_candidate,0,0)
            p_candidate = np.array(list(accumulate(p_candidate)))
            if p_candidate[-1]==0:
                add_i = sample(range(l),1)[0]
            else:
                add_i = find_lt(p_candidate,random()*p_candidate[-1])
            candidate_attributes = list(set(self.attributeNames).difference(set(MRS_new[add_i].keys())))
            for attr in candidate_attributes:
                candidate_values = list(set(self.attributeLevelIndeces[attr]))
                for value in candidate_values:
                    W_temp = W[add_i]*self.df.iloc[:,value]
                    TP,FP,TN,FN = getConfusion(W_temp,self.Y)
                    score_temp = float(TP)/(TP+FP+1)
                    # TP,FP,TN,FN = getConfusion(Yhat_temp[nindex],Y[nindex])
                    # score_temp = log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2)
                    # print 'score={},add_i={},attr={},value={}'.format(score_temp,add_i,attr,value)
                    if score_temp>=score_max:
                        score_max = score_temp
                        add_attribute = attr
                        add_value = value
                    # print 'score_max = {}, add_attribute = {},add_value = {},add_i = {}'.format(score_max,add_attribute,add_value,add_i)
        if add_i>=0:
            subdict = deepcopy(MRS_new[add_i])
            subdict[add_attribute].append(add_value)
            MRS_new.append(deepcopy(subdict))
            # if print_message:
            #     print 'Choose rule {} to add in jump'.format(add_i)
            #     print 'after jump:'
            #     self.printMRS(MRS_new)
        else:
            print('error!')

    def action_rmcondition(self,MRS_new,V,W,ex_dict,print_message):
        # # action1+=1
        # if random()<q:
        #     if print_message:
        #         print('select a RANDOM condition from a random rule')
        #     del_i = sample([i for i in range(len(MRS_new)) if W[i][ex]<lens[i]],1)[0]
        #     # print 'rules_new = {},del_i = {}'.format(rules_new,del_i)
        #     diff_attributes= [attr for attr in MRS_new[del_i].keys() if (not set(MRS_new[del_i][attr]).issubset(ex_dict[attr])) and MRS_new[del_i][attr]!=[]]
        #     # print 'diff_attributes = {}'.format(diff_attributes)
        #     del_attr = sample(diff_attributes,1)[0]
        # else:
        if print_message:
            print('selected the BEST condition to remove')
        score_max = -10000000000000
        del_val = -1
        self.MRS_new = MRS_new
        self.V = V
        self.W = W
        self.ex_dict = ex_dict
        for i in range(len(MRS_new)):
            # print('checking rule {}'.format(i))
            for att in MRS_new[i].keys():
                V_temp = deepcopy(V) # number of conditions satisfied in each rule
                W_temp = deepcopy(W) # indicator if a rule is satisfied
                if len(MRS_new[i][att])==1: # a condition has only one value
                    V_temp[i] = V[i] - self.df.ix[:,MRS_new[i][att][0]].values # assuming reduces MRS_new[i][att]
                    W_temp[i] = V_temp[i]==(len(MRS_new[i])-1)
                    Yhat_temp = (np.sum(W_temp,axis = 0)>0).astype(int)
                    TP,FP,TN,FN = getConfusion(Yhat_temp,self.Y)
                    score_temp =  float(TP)/(TP+FP+1)#TP*np.log(self.rho1) + FP*np.log(1-self.rho1)+ TN*np.log(self.rho2) + FN*np.log(1-self.rho2)
                    self.score_temp = score_temp
                    if score_temp>=score_max:
                        score_max = score_temp
                        del_attr = att
                        del_i = i
                        del_val = -1 # only one value
                        # print('{}: attr = {}, TP = {}, FP = {}, TN = {}, FN = {}, score = {}, score_max = {}'.format(i,att,TP,FP,TN,FN,score_temp,round(score_max,2)))
                else: 
                    for val in MRS_new[i][att]:
                        vals = [v for v in MRS_new[i][att] if v != val]
                        V_temp[i] = V[i] - (np.sum(self.df.iloc[:,vals],axis = 1)>0).astype(int).values
                        W_temp[i] = V_temp[i]==(len(MRS_new[i])-1)
                        Yhat_temp = (np.sum(W_temp,axis = 0)>0).astype(int)
                        TP,FP,TN,FN = getConfusion(Yhat_temp,self.Y)
                        score_temp =  float(TP)/(TP+FP+1)#TP*np.log(self.rho1) + FP*np.log(1-self.rho1)+ TN*np.log(self.rho2) + FN*np.log(1-self.rho2)
                        # print('{}: attr = {}, TP = {}, FP = {}, TN = {}, FN = {}, score = {}, score_max = {}'.format(i,att,TP,FP,TN,FN,score_temp,round(score_max,2)))
                        if score_temp>=score_max:
                            score_max = score_temp
                            del_attr = att
                            del_i = i
                            del_val = val
                        # score_temp = log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2)
        if del_val == -1: # either remove a condition or remove a value
            MRS_new[del_i].pop(del_attr)
            if len(MRS_new[del_i])==0:
                MRS_new.pop(del_i)
        else:
            MRS_new[del_i][del_attr] = [x for x in MRS_new[del_i][del_attr] if x != del_val]
        if print_message:
            print('action1: remove attribute {} from rule {}'.format(del_attr,del_i) )    

    def action_rmrule(self,MRS_new,W,print_message):
        # action4+=1
        # if trim==False and random()<q:
        #     if print_message:
        #         print('Select a RANDOM rule to remove')
        #     del_i = sample(indeces,1)[0]
        # else:
        if print_message:
            print('Select a BEST rule to remove')
        all_sum = np.sum(W,axis = 0)
        score_max = -1
        del_i = -1
        for i in range(len(MRS_new)):
            Yhat_temp= (all_sum - W[i]>0).astype(int)
            TP,FP,TN,FN  = getConfusion(Yhat_temp,self.Y)
            score_temp = float(TP)/(TP+FP+1)
            # score_temp = log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
            if score_temp>=score_max:
                score_max = score_temp
                del_i = i
        if del_i>=0:
            MRS_new.pop(del_i)
        if print_message:
            print('action4: remove rule {} from MRS'.format(del_i))

    def action_addrule(self,MRS_new,W,Yhat,ex,ex_dict,maxlen,q,print_message):
        # add a subset from ex_dict as a new rule:
        if print_message:
            print('Add a BEST rule')
        score_max = -10000
        neg_ind = np.where(Yhat==0)[0]
        select = list(set(np.where(self.supp>self.minsupp[-1])[0]).intersection(np.where(self.RMatrix[ex,:]==1)[0]))
        if len(select)>0:
            if random()<q:
                add_rule_ind = sample(select,1)[0]
            else: 
                Yhat_neg_index = np.where(Yhat==0)[0]
                add_rule_ind = select[0]
                for ind in select[1:]:
                    TP,FP,TN,FN = getConfusion(self.RMatrix[neg_ind,ind],self.Y[neg_ind])
                    score_temp = float(TP)/(TP+FP+1)
                    if score_temp>=score_max:
                        score_max = score_temp
                        add_rule_ind = ind
            new_dict = defaultdict(list)
            rule = self.rules[add_rule_ind]
            for condition in rule:
                try:
                    att,_ = condition.split('_')
                    new_dict[att]= [self.colindices[condition]]
                except:
                    print(condition)
            MRS_new.append(new_dict)
            # self.MRS_new_fromadd = [deepcopy(subdict) for subdict in MRS_new]
    
    def action_addcondition(self,MRS_new,ex_dict,W,print_message):
        # # action3+=1
        # if random()<q: # randomly select a rule to add
        #     if print_message:
        #         print('Select a RANDOM item to add to a random rule')
        #     add_i = sample(indeces,1)[0]
        #     add_attribute = sample(set(self.attributeNames).difference(set(MRS_new[add_i].keys())),1)[0]
        #     add_value = sample(set(self.attributeLevelIndeces[add_attribute]).difference(set(ex_dict[add_attribute])),1)[0]
        # else:
        # add a condition such that the rule set DOES NOT cover ex
        if print_message:
            print('Select the BEST condition to add')
        score_max = -1000000
        add_i = -1
        for i,subdict in enumerate(MRS_new):
            # InfoGain_i = compute_InfoGain(W[i],self.Y)
            if sum(W[i])>self.minsupp[-1]: # Contrapositive property
                candidate_attributes =[i for i in self.attributeNames if i not in subdict.keys()]
                for attr in candidate_attributes:
                    candidate_values = list(set(self.attributeLevelIndeces[attr]).difference(set(ex_dict[attr])))
                    for value in candidate_values:
                        # self.W = W
                        # self.i = i
                        TP,FP,TN,FN = getConfusion(W[i],self.Y)
                        score1 = float(TP)/(TP+FP+1)
                        Z = W[i]*self.df.iloc[:,value]
                        TP,FP,TN,FN = getConfusion(Z,self.Y)
                        score2 = float(TP)/(TP+FP+1) #log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
                        score_d = score2 - score1
                        if score_d>=score_max and TP+FP >self.minsupp[-1]:
                            score_max = score_d
                            add_attribute = attr
                            add_value = value
                            add_i = i
                            TP_max,FP_max,TN_max,FN_max = TP,FP,TN,FN
        if add_i>=0:
            if add_value not in MRS_new[add_i][add_attribute]:
                MRS_new[add_i][add_attribute].append(add_value)
            if print_message:
                print('action3 - add condition: add value {} to attribute {} in rule i = {}, score = {}, TP = {}, FP = {}, TN = {}, FN = {}'.format(add_value,add_attribute,add_i,score_max,TP_max,FP_max,TN_max,FN_max))

    def compute_prob(self, MRS_new):
        MRS_merged= self.merge(MRS_new)
        MRS_new_rlens = [len(subdict) for subdict in MRS_merged]
        MRS_new_len = len(MRS_new_rlens)
        W = np.zeros((len(MRS_merged),self.N)) # indicator, if an example satisfies a rule
        V = np.zeros((len(MRS_merged),self.N)) # how many conditions each example satisfies in a rule 
        rm_attributes_list = []
        for i,subdict in enumerate(MRS_merged):
            for att in subdict.keys():
                tmp = (np.sum(self.df.iloc[:,subdict[att]],axis = 1)>0).astype(int)
                V[i] += tmp
                if np.mean(tmp)==1:
                    rm_attributes_list.append((i,att))
            W[i] = (V[i]==MRS_new_rlens[i]).astype(int)
        for i,att in rm_attributes_list:
            MRS_merged[i].pop(att,None)
        Yhat = (np.sum(W,axis = 0)>0).astype(int)
        TP,FP,TN,FN = getConfusion(Yhat,self.Y)        
        N = len(MRS_merged)
        Ln = [len([item for sublist in subdict.values() for item in sublist]) for subdict in MRS_merged]
        Jn = [[0 if att not in subdict.keys() else len(subdict[att]) for att in self.attributeNames] for subdict in MRS_merged]
        prior_NumOfRules = log_gampoiss(N,self.alpha_M,self.beta_M)
        prior_NumOfItems= sum([log_gampoiss(l, self.alpha_L,self.beta_L) for l in Ln])
        prior_ChsItems = sum([log_dirmult(row,self.alpha) for row in Jn])
        likelihood_1 =  log_betabin(TP,TP+FP,self.alpha_1,self.beta_1)
        likelihood_2 = log_betabin(TN,FN+TN,self.alpha_2,self.beta_2)
        # # likelihood_1 = TP*np.log(self.rho1) + FP*np.log(1-self.rho1)
        # # likelihood_2 = TN*np.log(self.rho2) + FN*np.log(1-self.rho2)
        # likelihood_1 = 0
        # likelihood_2 = TP+TN
        pt_new = prior_NumOfRules + prior_NumOfItems + prior_ChsItems + likelihood_1 + likelihood_2
        return MRS_merged,[TP,FP,TN,FN],[prior_NumOfRules,prior_NumOfItems,prior_ChsItems ,likelihood_1,likelihood_2],Yhat,W,V,MRS_new_rlens,MRS_new_len

    def merge(self,MRS):
        MRS_merged = [deepcopy(subdict) for subdict in MRS]
        MRS_merged = sorted(MRS_merged,key = lambda x:len(x))
        self.MRS_merged = MRS_merged
        while True:
            rm_list = []
            for row_indexA in range(len(MRS_merged)):
                self.row_indexA = row_indexA
                attributeA = set(MRS_merged[row_indexA].keys())
                if row_indexA not in rm_list:
                    for row_indexB in range(len(MRS_merged)):
                        if row_indexB != row_indexA and row_indexB not in rm_list:
                            attributeB = set(MRS_merged[row_indexB].keys())
                            if attributeA.issubset(attributeB):
                                subset = 0
                                same = 0
                                for attr in attributeA:
                                    # self.MRS_merged = MRS_merged
                                    # self.row_indexB = row_indexB
                                    # self.attr = attr
                                    if set(MRS_merged[row_indexB][attr])<=set(MRS_merged[row_indexA][attr]):
                                        subset += 1
                                        if set(MRS_merged[row_indexB][attr])==set(MRS_merged[row_indexA][attr]):
                                            same += 1
                                if same==(len(attributeA)-1) and len(attributeA)==len(attributeB):
                                    # print('merging index_A = {}, index_B = {}'.format(row_indexA, row_indexB))                
                                    for attr in attributeA:
                                        MRS_merged[row_indexA][attr] = list(set(MRS_merged[row_indexA][attr]).union(set(MRS_merged[row_indexB][attr])))
                                    rm_list.append(row_indexB)           
                                elif subset == len(attributeA):
                                    # print('deleting index_A = {}, index_B = {}'.format(row_indexA, row_indexB)) 
                                    rm_list.append(row_indexB) 
                                    # model.printMRS(MRS_merged)
                        row_indexB += 1
                    rm = []
                #     print('A = {}, B = {}, len = {}'.format(row_indexA,row_indexB,len(MRS_merged)))
                    for attr in MRS_merged[row_indexA].keys():
                        values = MRS_merged[row_indexA][attr]
                        self.values = values
                        self.MRS_merged = MRS_merged
                        if  sum(np.sum(self.df.iloc[:,values],axis = 1)>0) == self.N:
                        # if len(MRS_merged[row_indexA][attr])==self.attributeLevelNum[attr]:
                            rm.append(attr)
                    for attr in rm:
                        MRS_merged[row_indexA].pop(attr, None)
                            # del MRS_merged[row_indexA][attr]
                    if len(MRS_merged[row_indexA])==0:
                        MRS_merged.pop(row_indexA)
                    row_indexA += 1
            MRS_merged = [subdict for ind,subdict in enumerate(MRS_merged) if ind not in rm_list]
            if len(rm_list)==0:
                break

        return MRS_merged

    def printMRS(self,MRS):
        for i,subdict in enumerate(MRS):
            rule = 'rule {}:'.format(i)
            for key in subdict.keys():
                rule += '('+str(key)+':'
                values = [self.itemNames[value].split('_')[1] for value in subdict[key]]
                values = ' or '.join(values)
                rule = rule + values + '),'
            print(rule)

    # def printex(self,ex_dict):
    #     for key in ex_dict.keys():
    #         string =  '('+key+':'
    #         for value in ex_dict[key]:
    #             string += self.itemNames[value].split('_')[1] +'+'
    #         string += '),'
    #         print string

    def print_rules(self, rules_max):
        for rule_index in list(rules_max):
            rule =[self.itemNames[ind] for ind in list(rule_index)]
            print(rule)

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    else:
        return 0


def log_betabin(k,n,alpha,beta):
    import math
    try:
        Const =  math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print('alpha = {}, beta = {}'.format(alpha,beta))
    if isinstance(k,list) or isinstance(k,np.ndarray):
        if len(k)!=len(n):
            print('length of k is %d and length of n is %d'%(len(k),len(n)))
            raise ValueError
        lbeta = []
        for ki,ni in zip(k,n):
            # lbeta.append(math.lgamma(ni+1)- math.lgamma(ki+1) - math.lgamma(ni-ki+1) + math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
            lbeta.append(math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
        return np.array(lbeta)
    else:
        return math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const


def predict_MRS(MRS,df):
    Z = np.zeros(len(df))
    for i,subdict in enumerate(MRS):
        tmp = np.zeros(len(df))
        for att in subdict.keys():
            tmp += (np.sum(df.iloc[:,subdict[att]],axis = 1)>0).astype(int)
        Z += (tmp == len(subdict)).astype(int)
    Yhat = (Z>0).astype(int)
    return Yhat

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
