# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:27:42 2020

@author: ajenul
"""
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import matthews_corrcoef

my_reg_params = [0.1,1,10]
my_l1_params = [0, 0.75, 0.9, 1]
testsize_range = (0.25, 0.25)

def parameter_selection(data, 
                        labels,
                        my_reg_params, 
                        my_l1_params, 
                        n_splits, 
                        testsize_range):

    """
    select best parameters for elastic net within RENT
    for CLASSIFICATION!
    """
        
    if len(np.shape(labels)) == 2:
        labels = labels.squeeze()
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle = True)
    
    scores_df = pd.DataFrame(np.zeros, index=my_l1_params, columns=my_reg_params)
    zeroes_df = pd.DataFrame(np.zeros, index=my_l1_params, columns=my_reg_params)
    
    
    def run_parallel(l1):
        """
        Parallel computation of for loops. Parallelizes the number of tt_splits
        as this is the parameter with most varying values.
        
        INPUT
        -----
        tt_split: range of train-test splits
        
        OUTPUT
        ------
        None 
        """
    
        
        for reg in my_reg_params:
            scores = list()
            zeroes = list()
            for train, test in skf.split(data, labels):
                
                sc = StandardScaler()
                train_data = sc.fit_transform(data.iloc[train,:])
                train_target = labels[train]
                test_data_split = sc.transform(data.iloc[test,:])
                test_target = labels[test]
                sgd = LR(penalty="elasticnet", C=reg, solver="saga", 
                         l1_ratio=l1, random_state=0)
                sgd.fit(train_data, train_target)
                
                params = np.where(sgd.coef_ != 0)[1]
    
                
                if len(params) == 0:
                    scores.append(np.nan)
                    zeroes.append(np.nan)
                else:
                    zeroes.append((len(data.columns)-len(params))\
                                  /len(data.columns))
                    
                    sc = StandardScaler()
                    train_data_1 = sc.fit_transform(train_data[:,params])
                    test_data_1 = sc.transform(test_data_split[:, params])
                    model = LR(penalty='none', max_iter=8000, solver="saga",
                               random_state=0).\
                            fit(train_data_1,train_target)
                    scores.append(matthews_corrcoef(test_target, 
                                                    model.predict(test_data_1)))
                            
            scores_df.loc[l1,reg] = np.nanmean(scores)
            zeroes_df.loc[l1,reg] = np.nanmean(zeroes)
        
        #return(scores_df, zeroes_df)
    
    Parallel(n_jobs=-1, verbose=0, backend="threading")(
         map(delayed(run_parallel), my_l1_params))  
    
    print(scores_df)
    #return(scores_df, zeroes_df)
    
    print('hallo')
    normed_scores = (scores_df-np.nanmin(scores_df.values))\
    /(np.nanmax(scores_df.values)-np.nanmin(scores_df.values))
    normed_zeroes = (zeroes_df-np.nanmin(zeroes_df.values))\
    /(np.nanmax(zeroes_df.values)-np.nanmin(zeroes_df.values))
    
    combi = (normed_scores ** -1 + normed_zeroes ** -1) ** -1
    best_combi_row, best_combi_col  =np.where(combi == np.nanmax(combi.values))
    best_l1 = combi.index[np.nanmax(best_combi_row)]
    best_C = combi.columns[np.nanmin(best_combi_col)]
    
    return(best_C, best_l1)