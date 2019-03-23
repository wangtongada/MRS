from pandas.io.parsers import read_csv
import numpy as np
import pandas as pd


def code_intervals_nlevel(df,collist,Nlevel):
    for col in collist:
        df[col+'_missing'] = isnan(df[col]).astype(int)
        index = np.where(~isnan(df[col]))
        for q in range(Nlevel):
            l = df[col].quantile(float(q)/Nlevel)
            u = df[col].quantile(float(q+1)/Nlevel)
            df[col+'_'+str(q)] = 0
            df[col+'_'+str(q)].iloc[index] = ((df[col].iloc[index] >=l) & (df[col].iloc[index]<u)).astype(int).values
    df.drop(collist,axis = 1, inplace = True) 

Xtrain = read_csv('recidivism_train.csv',header = 0, sep = ',')
Xtest = read_csv('recidivism_test.csv',header = 0, sep = ',')
Y = Xtrain['Y']
Xtrain.drop('Y',axis = 1, inplace = True)
numericals = []
for col in Xtrain.columns:
    if '_' not in col:
        numericals.append(col)
thresholds = defaultdict(list)
for col in numericals:
    thresholds[col] = [Xtrain[col].quantile(float(q)/Nlevel) for q in range(Nlevel)]
code_intervals_thresholds(Xtrain,numericals,thresholds)

model = MRS(Xtrain,Y) 
model.generate_rules(supp,maxlen,Nrules,method = 'fpgrowth',criteria = 'precision')
model.set_parameters(10,1,10,1,1,betaM)
grs,maps = model.train(Niteration,0.2,False)
