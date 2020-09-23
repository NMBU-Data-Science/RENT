# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:35:24 2020

@author: ajenul
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:14:31 2020

@author: ajenul
"""

import hoggorm as ho
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations, combinations_with_replacement
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.stats import t




#==============================================================================
# Define class
#==============================================================================


class RENT:
    """
    This class carries out repeated elastic net feature selection on a given
    binary classification dataset. Feature selection is done on multiple train 
    test splits. The user can initiate interactions between features that are
    included in the dataset and as such introduce non-linearities.
    
    INPUT
    -----
    
    data: numpy array
    
    target: numpy array
    
    feat_names: list holding feature names
    
    scale: boolean, default=True
    
    reg_params: list holding regularisation parameters for model
    
    l1_params: list holding ratio between l1 and l2 penalty
    
    poly: str, options: 'OFF', no feature interaction
                        'ON', (includes squared of features)
                        'ON_only_interactions', (only interactions, 
                                                 no squared features)
    
    testsize_range: tuple (low, high) range of random proportion of dataset to
    include in test set,
                    low and high are floats between 0 and 1, 
                    default (0.2, 0.6)
    
    scoring: str, options: 'accuracy', 'f1', 'precision', 'recall', 'matthews'
    
    clf: str, options: 'logreg' for logistic regression
                       'linearSVC' for linear support vector classifier
                       'RM' for linear regression
    
    num_tt: int, number of unique train-test splits 

    num_w_init: int, number of unique weight intialisations                  
    verbose: int, track running if greater than 
    
    OUTPUT
    ------
    None 
    """

    def __init__(self, data, target, feat_names=[],
                 scale=True, reg_params=[1], poly='OFF',
                 testsize_range=(0.2, 0.6),
                 scoring='accuracy', clf='logreg',
                 num_tt=5, num_w_init=10, l1_params = [0.6],
                 verbose = 0):
        
        # Print parameters for checking
        print('Dim data:', np.shape(data))
        print('Dim target', np.shape(target))
        print('reg param C:', reg_params)
        print('l1_params:', l1_params)
        print('num TT splits:', num_tt)
        print('num weight inits:', num_w_init)
        print('data type:', type(data))
        print('verbose:', verbose)

        

        # Define all objects needed later in methods below
        self.target = target
        self.reg_params = reg_params
        self.l1_params = l1_params
        self.num_tt = num_tt
        self.num_w_init = num_w_init
        self.feat_names = feat_names
        self.scoring = scoring
        self.clf = clf
        self.testsize_range = testsize_range
        self.verbose = verbose
        
        
        # Check if data is dataframe and add index information
        if isinstance(data, pd.DataFrame):
            self.indices=data.index
        else:
            self.indices=list(range(data.shape[0]))


        # If no feature names are given, then make some
        if len(self.feat_names) == 0:
            print('No feature names given. Generating some ...')
            
            for ind in range(1, np.shape(data)[1] + 1):
                #print('f' + str(ind))
                self.feat_names.append('f' + str(ind))
        
                
        # Extend data if poly was set to 'ON' or 'ON_only_interactions'
        if poly == 'ON':
            polynom = PolynomialFeatures(interaction_only=False, \
                                         include_bias=False)
            self.data = polynom.fit_transform(data)
            polynom_comb = list(combinations_with_replacement(self.feat_names,\
                                                              2))
            polynom_feat_names = []
            
            # Construct a new name for squares and interactions
            for item in polynom_comb:
                if item[0] == item[1]:
                    name = item[0] + '^2'
                else:
                    name = item[0] + '*' + item[1]
                polynom_feat_names.append(name)
                
            flist=list(self.feat_names)
            flist.extend(polynom_feat_names)
            self.feat_names = flist
            self.data = pd.DataFrame(self.data)
            self.data.index=self.indices
            self.data.columns = self.feat_names
        
        elif poly == 'ON_only_interactions':
            polynom = PolynomialFeatures(interaction_only=True,\
                                         include_bias=False)
            self.data = polynom.fit_transform(data)

            #polynom_comb = list(combinations(['A', 'B', 'C'], 2))
            polynom_comb = list(combinations(self.feat_names, 2))
            polynom_feat_names = []

            # Construct a new name for squares and interactions
            for item in polynom_comb:
                name = item[0] + '*' + item[1]
                polynom_feat_names.append(name)
                
            flist = list(self.feat_names)
            flist.extend(polynom_feat_names)
            self.feat_names = flist
            self.data = pd.DataFrame(self.data)
            self.data.index = self.indices
            self.data.columns = self.feat_names
        
        elif poly == 'OFF':
            self.data = pd.DataFrame(data)
            self.data.index=self.indices
            self.data.columns = self.feat_names
        
        else:
            print('Value for paramter "poly" not regcognised.')


    def run_parallel(self,tt_split):
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

        # Loop through all C values in reg_params
        for C in self.reg_params:
            # Loop through requested number of tt splits
            for l1 in self.l1_params:
                
                if self.clf == "RM":
                    X_train, X_test, y_train, y_test = train_test_split(
                          self.data, self.target, 
                          test_size=self.random_testsizes[tt_split],
                          random_state=tt_split)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                          self.data, self.target, 
                          test_size=self.random_testsizes[tt_split],
                          stratify=self.target, random_state=tt_split)
                
                self.train_sets.append(X_train)
                self.X_test = X_test
                # Initialise standard scaler and compute mean and STD from 
                # training data
                sc = StandardScaler()
                sc.fit(X_train)
                        
                # Transform (standardise) both X_train and X_test with mean 
                # and STD from
                # training data
                X_train_std = sc.transform(X_train)
                X_test_std = sc.transform(X_test)
                if self.verbose > 0:
                    print('l1 = ',l1,'C = ', C, ', TT split = ', tt_split)

                # Loop through requested number of w_init
                for w_init in range(self.num_w_init):
                    
                    if self.clf == 'logreg':
                        # Trian a logistic regreission model
                        model = LogisticRegression(solver='saga', 
                                                C=C,
                                                penalty='elasticnet',
                                                l1_ratio=l1,
                                                n_jobs=-1,
                                                max_iter=5000,
                                                random_state=w_init).\
                                                fit(X_train_std, y_train)
                        
                    elif self.clf == 'linSVC':
                        model = LinearSVC(penalty='l1',
                                        C=C,
                                        dual=False,
                                        max_iter=8000,
                                        random_state=w_init).\
                                        fit(X_train_std, y_train)
                    
                    elif self.clf == "RM":
                        model = ElasticNet(alpha = 1/C, l1_ratio = l1,
                                           max_iter=5000, random_state=w_init, \
                                           fit_intercept=False).\
                                           fit(X_train_std, y_train)

                    # Get all weights (coefficients). Those that were selected
                    # are non-zero, otherwise zero
                    #print(logreg.coef_)
                    mod_coef = model.coef_
                    self.mcf = model.coef_
                    if len(np.shape(mod_coef)) ==1:
                        mod_coef = mod_coef.reshape(1, len(mod_coef))
                    self.weight_dict[(l1, C, tt_split, w_init)] = mod_coef
                    self.weight_list.append(mod_coef)
                    
                    if self.clf =="RM":
                        score = r2_score(y_test, model.predict(X_test_std))
                    # Check which metric to use
                    else:
                        if self.scoring == 'accuracy':
                            y_test_pred = model.predict(X_test_std)
                            score = model.score(X_test_std, y_test)
                        
                        elif self.scoring == 'f1':
                            y_test_pred = model.predict(X_test_std)
                            score = f1_score(y_test, y_test_pred)
                        elif self.scoring == 'precision':
                            y_test_pred = model.predict(X_test_std)
                            score = precision_score(y_test, y_test_pred)
                        elif self.scoring == 'recall':
                            y_test_pred = model.predict(X_test_std)
                            score = recall_score(y_test, y_test_pred)
                        elif self.scoring == "matthews":
                            y_test_pred = model.predict(X_test_std)
                            score = matthews_corrcoef(y_test, y_test_pred)
                        
                        # In addition store all scores for different evaluation
                        # metrics.
                        y_test_pred = model.predict(X_test_std)
                        sf = f1_score(y_test, y_test_pred)
                        sf_inv = f1_score((1-y_test), (1-y_test_pred))
                        sm = matthews_corrcoef(y_test, y_test_pred)
                        sa = model.score(X_test_std, y_test)
                        
                        sc = [sa, sf, sf_inv, sm]
                        self.sc[(l1, C, tt_split, w_init)] = sc
                    
                        sc_d = np.transpose(pd.DataFrame(sc))
                        sc_d.columns = ['acc','f1','f1 inv', 'matthews']
                        sc_d.index = [str(C)]
                        self.sc_pd =self.sc_pd.append(sc_d)
                    
                    self.score_all[(l1, C, tt_split, w_init)] = score
                    self.score_list_all.append(score)
                    self.score_dict[(l1, C, tt_split, w_init)] = score
                    self.score_list.append(score)
                    
                    # Collect true values and predictions in dictionary
                    if(self.clf != "RM" and self.clf != "linSVC"):
                        res_df = pd.DataFrame({"y_test":y_test, \
                                               "y_pred": y_test_pred})
                        res_df.index=X_test.index
                    
                        self.pred_dict[(l1, C, tt_split, w_init)] = res_df
                        
                    # calculate predict_proba for current train/test and weight
                    # initialization
                        res_df = pd.DataFrame(
                                {"y_test":y_test, "y_pred": y_test_pred})
                        res_df.index=X_test.index
                        self.pred_dict[(C,l1,tt_split,w_init)] = res_df
                        self.p[(C,l1,tt_split,w_init)] = pd.DataFrame( \
                               model.predict_proba(X_test_std), index = \
                                    X_test.index)
                    
    def pred_proba(self):
        """
        This method calculates the prediction probabilities when the classifier
        is logistic regression.
         
        
        INPUT
        -----
        None
        
        OUTPUT
        ------
        <dict> # is a dictionary
            A dictionary holding logistic regression prediction of class 1
            of each parameter combination.
            Dictionary key: (C, l1_ratio)
        """
        self.rdict = {}
        for C in self.reg_params:
            for l1 in self.l1_params:
                count =  0
                vec = pd.DataFrame(np.nan, index= self.indices, \
                                   columns = ["remove"])
                for k in self.p.keys():
                   
                    if k[0] == C and k[1] == l1:
                        vec.loc[self.p[k].index,count] = \
                        self.p[k].iloc[:,1].values
                        count = count+1
                        
                vec = vec.iloc[:,1:]
                
                self.rdict[(C, l1)] = vec
        return self.rdict
    
    def run_analysis(self):
        """
        This method trains C * num_tt * num_w_init models in total. The number
        of models using the same hyperparamter is num_tt * num_w_init .
        For each model elastic net regularisation is applied for variable 
        selection.
         
        
        INPUT
        -----
        None
        
        OUTPUT
        ------
        None 
        """
        np.random.seed(0)
        self.random_testsizes = np.random.uniform(self.testsize_range[0],
                                                  self.testsize_range[1], 
                                                  self.num_tt)
        
        # Initiate a dictionary holding coefficients for each model. Keys are
        # (C, num_tt, num_w_init)
        self.weight_dict = {}
        
        self.sc = {}
        self.sc_pd = pd.DataFrame(columns = ['acc','f1','f1 inv', 'matthews'])
        # Initiate a dictionary holding computed performance metric for each
        # model. Kes are (C, num_tt, num_w_init)
        self.score_dict = {}
        self.score_all = {}
        # Initiate a dictionary holding computed performance metric for each
        # model. Kes are (l1_param, C, num_tt, num_w_init)
        self.score_l1_dict = {}

        
        # Collect all coefs in a list that will be converted to a numpy array
        self.weight_list = []
        
        # Collect all coefs in a list that will be converted to a numpy array
        self.score_list = []
        self.score_list_all =[]
        # Initialize a dictionary to predict incorrect labels
        self.pred_dict = {}
        
        # store
        self.p = {}
        
        # pred_proba help function
        self.pred_proba_help = pd.DataFrame(np.nan, index=self.indices,\
                                            columns=['1'])
        
        
        

        # Count how often each object is used for testing and how often 
        # it is classified wrong
        self.incorrect_labels = pd.DataFrame({"nr.test":np.repeat\
                                              (0, np.shape(self.data)[0]),
                                              "class":self.target,
                                              "nr.incorrect":np.repeat\
                                              (0, np.shape(self.data)[0])})
        self.incorrect_labels.index=self.indices

        # Collect training data across all train-test splits
        self.train_sets = []

        # timestop
        start = time.time()
        # Call parallelization function 
        Parallel(n_jobs=-1, verbose=0, backend="threading")(
             map(delayed(self.run_parallel), range(self.num_tt)))  
        ende = time.time()
        self.runtime = '{:5.3f}s'.format(ende-start)
            
    def get_scores_summary_by_regParam(self, showall = False):
        """
        Returns the average of metric score across all 
        train-test splits and weight initialisations for each pair of (l1,C).
        
        INPUT
        -----
        showall: True if average score of all evaluation metrics shall be shown.
        
        OUTPUT
        ------
        <pandas DataFrame>: Dataframe containing average scores for the chosen
        evaluation metrix for each parameter combination.
        If showall is True, returns a dictionary.
        """

        if showall == True:
            scores = {}
            for l1 in self.l1_params:
                for C in self.reg_params:
                    arr = np.empty((0,4), int)
                    for k in self.sc.keys():
                        if k[0] == l1 and k[1] ==C:
                            arr = np.vstack((arr,self.sc[k]))
                            scores[(l1,C)] = np.transpose(pd.DataFrame(
                                    np.apply_along_axis(np.mean,0,arr)))
                            scores[(l1,C)].columns =  \
                            ['acc','f1','f1 inv', 'matthews']
            return scores
        else:
            result_list=[]
            for l1 in self.l1_params:
                for C in self.reg_params:
                    spec_result_list = []
                    for k in self.score_dict.keys():
                        if k[0] == l1 and k[1] ==C:
                            spec_result_list.append(self.score_dict[k])
                    result_list.append(spec_result_list)
            
        
            means=[]
            for r in range(len(result_list)):
                means.append(np.mean(result_list[r]))
    
            summary_df = pd.DataFrame(np.array(means).reshape(\
                                      len(self.l1_params), \
                                      len(self.reg_params)), \
            index= self.l1_params, columns = self.reg_params)
        
            return summary_df
    
    
    def get_average_zero_features(self):
        """
        Returns the average of features set to 0 for each pair (l1,C). 
        
        INPUT
        -----
        None
        
        OUTPUT
        ------
        <pandas DataFrame>: 
        """
        result_list = pd.DataFrame(index = self.l1_params,\
                                   columns=self.reg_params)
        for l1 in self.l1_params:
            for C in self.reg_params:
                count=0
                for tt_split in range(self.num_tt):
                    for w_init in range(self.num_w_init):
                        
                        nz = \
                        len(np.where(pd.DataFrame(self.weight_dict[(l1, \
                                                                     C,\
                                                                     tt_split,\
                                                                     w_init)\
    ])==0)[0])
                        count = count + nz/len(self.feat_names)
                count = count / (self.num_tt * self.num_w_init)
                result_list.loc[l1, C] = count
        return result_list
    
    def zero_parameters(self, vec):
        """
        Returns mean and standard deviation of feature-model weights without
        those where the weight is 0.
    
        INPUT
        ----
        a vector array (must not be a list!)
        
        OUTPUT
        ------
        list of mean and standard deviation, percentage of elements != 0,
        mean of absolute weights, number of elements != 0 
        """
        # Count of weights != 0
        element_count = len(np.where(vec!=0)[0])
        # Mean of elements != 0
        mean_nzero = sum(vec)/element_count
        # Standard deviation of elements != 0, ddof = 1
        std_nzero = np.sqrt(sum((vec[np.where(vec!=0)[0]]-mean_nzero)**2)\
                            /(element_count-1))
        # Percentage how often feature is selected
        perc = element_count/len(vec)
        # Mean of absolute elements != 0
        mean_abs_nzero = sum(abs(vec))/element_count
        
        return(mean_nzero, std_nzero, perc, mean_abs_nzero, element_count)
        
        
            
    def get_all_weights_summary(self):
            """
            Provides a summary of weights across all computed models.
            
            INPUT
            -----
            None
            
            OUTPUT
            ------
            <DataFrame> containing summary on weights for each feature across 
                        all computed models
                        
                row 1: mean weight
                row 2: mean of absolute weights
                row 3: standard deviation of weights
                row 4: number of times feature weights was non-zero
                row 5: percentage of how often weights were non-zero
                row 6: abs(row 1) / row 2
                row 7: abs(row 1) / row 3
            """
            
            # Collect all weights in an array
            self.weight_arr = np.vstack(self.weight_list)
            
            # Compute results based on weights
            means = np.mean(self.weight_arr, axis=0)
            stds = np.std(self.weight_arr, axis=0)
            abs_means = np.mean(np.abs(self.weight_arr), axis=0)
            frac_mean = abs(means) / abs_means
            mean_by_std = abs(means / stds),
            counts_non_zero = np.count_nonzero(self.weight_arr, axis=0)
            perc_non_zero = counts_non_zero / len(self.weight_list) * 100
            mean_nzero, std_nzero, p, mean_abs_nzero, m = \
            np.apply_along_axis(self.zero_parameters\
                                           , 0, self.weight_arr)
            def sign_vote(x):
                return np.abs(np.sum(np.sign(x)))/len(x)
            sig_criterium = np.apply_along_axis(sign_vote,0,self.weight_arr)
            t_test_with_zero = t.cdf(
                    abs(means / np.sqrt((stds**2)/len(self.weight_arr))), \
           (len(self.weight_arr)-1))
            
            # Construct a dataframe thaw holds the results compted above
            summary = np.vstack([means, abs_means, stds, 
                                 counts_non_zero, perc_non_zero,
                                 frac_mean, mean_by_std, mean_nzero, 
                                 std_nzero, sig_criterium, t_test_with_zero])
            summary_df = pd.DataFrame(summary)
            summary_df.index = ['mean(w)', 
                                'mean(abs(w))', 
                                'std(w)', 
                                'count non-zero', 
                                'perc non-zero',
                                'abs(mean(w)) / mean(abs(w))',
                                'abs(mean(w) / std(w))',
                                'mean_non-zero',
                                'std_non-zero',
                                'sig_criterium',
                                't_test_with_zero']
            
            # Insert feature names in data frame
            summary_df.columns = self.feat_names
            
            # Make plot that shows percentages
            fig = plt.figure(figsize=(10, 7))
            (markers, stemlines, baseline) = plt.stem(perc_non_zero,\
            use_line_collection=True)
            plt.setp(markers, marker='o', markersize=4, color='orange', 
                     markeredgecolor="orange", markeredgewidth=0)
            plt.setp(stemlines, color='slateblue', linewidth=0.5)
            plt.show()
            
            # Return the summary dataframe
            return summary_df    
    
    def get_all_weights_dict(self):
        """
        Returns weights (coefficients) of all computed models in a dictionary. 
    
        
        INPUT
        ----
        None
        
        OUTPUT
        ------
        self.coef_dict: <dictionary>
            A dictionary holding weights of each computed model.
            Dictionary key: (C, num_tt, num_w_init)
        """
        return self.weight_dict
    
    
    def get_all_scores_dict(self):
        """
        Returns score (performance metric) of all computed models in a 
        dictionary. 
        
        INPUT
        ----
        None
        
        OUTPUT
        ------
        <dictionary>
            A dictionary holding score of each computed model.
            Dictionary key: (C, num_tt, num_w_init)
        """
        return self.score_dict
    
    def get_all_weights_df(self):
        """
        Returns weights (coefficients) of all computed models in a pandas
        dataframe. 
          
        INPUT
        ----
        None
        
        OUTPUT
        ------
        <pandas dataframe>
            A dataframe holding weights of each computed model.
            Dictionary key: (C, num_tt, num_w_init)
        """
        self.weight_arr = np.vstack(self.weight_list)
        weight_df = pd.DataFrame(self.weight_arr)
        weight_df.index = list(self.weight_dict.keys())
        weight_df.columns = self.feat_names
        
        return weight_df
    
    
    def get_all_scores_df(self):
        """
        Returns score (performance metric) of all computed models in a 
        pandas dataframe. All results are stored in one column.
        
        INPUT
        ----
        None
        
        OUTPUT
        ------
        <pandas dataframe>
            A dictionary holding score of each computed model.
            Dictionary key: (C, num_tt, num_w_init)
        """
        
        if self.scoring =="all":
            print("Has to be adjusted for the scoring=all case")
        else:
            self.score_arr = np.vstack(self.score_list)
            scores_df = pd.DataFrame(self.score_arr)
            scores_df.index = list(self.weight_dict.keys())
            scores_df.columns = ['performance metric']
        
            return scores_df
    
    def get_all_scores_by_regParam_df(self):
        """
        Returns score (performance metric) of all computed models in a 
        pandas dataframe. Results are stored in columns, where each column
        represents one regularisation parameter C.
        
        INPUT
        ----
        None
        
        OUTPUT
        ------
        <pandas dataframe>
            A dictionary holding score of each computed model.
            Dictionary key: (num_tt, num_w_init)
        """
        # I don't think that we need this method.. and with varying l1 parameter
        # it would be 3-dimensional
        print("not done yet")
        # Loop through all regularisation parameters        
#        result_list = []
#        for C in self.reg_params:
#            #print(C)
#            spec_result_list = []
#            
#            # Loop through all train-test split seeds
#            for tt_split in range(self.num_tt):
#                
#                # Loop through all weight initialsation seeds
#                for w_init in range(self.num_w_init):
#                    spec_result_list.append(self.score_dict[(C, tt_split,\
#                                                             w_init)])
#        
#            result_list.append(spec_result_list)
#        
#        # Collect all results in a numpy array
#        score_arr = np.transpose(np.vstack(result_list))
        
        
# =============================================================================
#         here starts code where l1 is just one value
# =============================================================================
#        result_list = []
#        for C in self.reg_params:
#            spec_result_list = []
#            for k in self.score_dict.keys():
#                if k[0] == C:
#                    spec_result_list.append(self.score_dict[k])
#            result_list.append(spec_result_list)
#        #Collect all results in a numpy array
#        score_arr = np.transpose(np.vstack(result_list))
#        # Make dataframe based that also includes feature names and row names
#        # inidicating for which train-test split and weight initialisation
#        # seed scores were computed.
#        score_df = pd.DataFrame(score_arr)
#        index_list = []
#         
##        index_list = [(tt_split, w_init) for tt_split in range(self.num_tt) \
##                      for w_init in range(self.num_w_init)]
#        
#
#
#        for k in self.score_dict.keys():
#            if (k[2],k[3]) not in index_list:
#                index_list.append( (k[2],k[3]))
#        
#        # Insert feature names in data frame
#        reg_param_names = []
#        for reg_param in self.reg_params:
#            reg_param_names.append(str(reg_param))
#        score_df.columns = reg_param_names
#        score_df.index = index_list
#        
#        return score_df
    
    def get_spec_weights_summary(self, 
                                 reg_param,
                                 l1_param,
                                 cutoff_perc=0.9, 
                                 cutoff_means_ratio=0.9, 
                                 cutoff_mean_std_ratio=1,
                                 feature_size = None,
                                 sel_approach = "old",
                                 plot = True):
        """
        Provides a summary of weights across all computed models for a 
        specific regularisation paramter chosen by user.
        
        INPUT
        -----
        reg_param: <int> values for regularisation paramter
        
        cutoff_perc: <int> or <float> 
            Cutoff critera for feature selection. Minimum Frequency of how 
            often a feature must have been selected across all models
            for given regularisation parameter. Choose value between 0 and 
            1 (in %).
        
        cutoff_means_ratio: <float>
            Cutoff criteria for feature selection. For a feature to be selected
            criteria 2 must be higher than cutoff_means_ratio.
        
        cutoff_mean_std_ratio: <float>
            Cutoff criteria for feature selection. For a feature to be selected
            criteria 3 must be higher than cutoff_mean_std_ratio.
        feature_size: <int>: number of features that shall be seleted (criteria
                                                            must be formualted)
        sel_approach: "old" or "new" dependent on old criteria or new criteria
        with sigmoid function and t-test
            
        
        OUTPUT
        ------
        <tuple> A tuple holding two pandas dataframes and one numpy array. The 
        first holds a summary on the weights (as in get all weights summary), 
        however this time only for the selected regularisastion paramter. The
        second dataframe holds original data of only selected features as set 
        by cutoff criteria. The array contains the position of the selected 
        variables.
                
        (pandas DataFrame 1, pandas DataFrame 2, numpy array)
        """
        
        # Check given C is contained in list of C's


        if reg_param not in self.reg_params:
            print('No computation for this regularization parameter available.')
            return None
        elif l1_param not in self.l1_params:
            print('No computation for this l1 parameter available.')
            return None
        
        
        
        else:
            self.feature_size = feature_size
            spec_weight_list = []
            
            # Loope through all train-test splits
            for tt_split in range(self.num_tt):
                
                # Loop through all weight initialisations
                for w_init in range(self.num_w_init):
                    spec_weight_list.append(self.weight_dict[(l1_param,\
                                                              reg_param,\
                                                              tt_split,\
                                                              w_init)])
        
            spec_weight_arr = np.vstack(spec_weight_list)  
            
            # calculate correlation matrix and the maximal correlation for each          
            # variable and return value with higest absolute value
            def max_min(correlations):
                ma = max(correlations)
                mi = min(correlations)
                
                if abs(ma) >= abs(mi):
                    return(ma)
                else:
                    return(mi)
            
            corr_mat = np.array(self.data.corr())
            np.fill_diagonal(corr_mat, 0)
            # Highest correlation for each feature with any other feature
            corr_list = list(np.round(np.apply_along_axis(max_min, 0,\
                                                          corr_mat), 2))
            
            # Compute results based on weights
            counts_non_zero = np.count_nonzero(spec_weight_arr, axis=0)
            perc_non_zero = counts_non_zero / len(spec_weight_list) * 100
            
            # Statistics based on all weights
            means = np.mean(spec_weight_arr, axis=0)
            stds = np.std(spec_weight_arr, axis=0)
            abs_means = np.mean(np.abs(spec_weight_arr), axis=0)
            frac_mean = abs(means) / abs_means
            mean_by_std = abs(means / stds)
            
            # Statistics based on non-zero weights
            mean_nzero, std_nzero, perc, mean_abs_nzero, element_count = \
            np.apply_along_axis(self.zero_parameters, 0, spec_weight_arr)
            frac_mean_nzero = abs(mean_nzero)/mean_abs_nzero
            mean_by_std_nzero = abs(mean_nzero)/std_nzero
#            highest_corr = corr_list

            # signum function
            def sign_vote(x):
                return np.abs(np.sum(np.sign(x)))/len(x)
            sig_criterium = np.apply_along_axis(sign_vote,0,spec_weight_arr)
            
            # t-statistic
            t_test = t.cdf( abs(
                    (mean_nzero / np.sqrt((std_nzero**2)/element_count))), \
                       (element_count-1))
            t_test_with_zero = t.cdf(
                    abs(means / np.sqrt((stds**2)/len(spec_weight_list))), \
           (len(spec_weight_list)-1))
            
            # Construct a dataframe thaw holds the results compted above
            summary = np.vstack([means, abs_means, stds, 
                                 counts_non_zero, perc, frac_mean,
                                 mean_by_std, mean_nzero, std_nzero,
                                 mean_abs_nzero, frac_mean_nzero, 
                                 mean_by_std_nzero,
                                 sig_criterium, t_test, corr_list,
                                 t_test_with_zero])
            
            summary_df = pd.DataFrame(summary)
            summary_df.index = ['mean(w) (C={0})'.format(reg_param), 
                                'mean(abs(w)) (C={0})'.format(reg_param), 
                                'std(w) (C={0})'.format(reg_param), 
                                'count non-zero (C={0})'.format(reg_param), 
                                'perc non-zero (C={0})'.format(reg_param),
                                'abs(mean(w)) / mean(abs(w))'.format(reg_param),
                                'abs(mean(w) / std(w))'.format(reg_param),
                                'mean_non-zero (C={0})'.format(reg_param),
                                'std_non-zero (C={0})'.format(reg_param),
                                'mean(abs) non-zero (C={0})'.format(reg_param),
                                'abs(mean(w)) / mean(abs(w)) non-zero',
                                'abs(mean(w) / std(w)) non-zero',
                                'sig_criterium',
                                't_test',
                                'highest correlation'.format(reg_param),
                                't_test_with_zero']
            
            # Insert feature names in data frame
            summary_df.columns = self.feat_names
            
            # Make plot that shows percentages
            if plot == True:
                fig = plt.figure(figsize=(10, 7))
                (markers, stemlines, baseline) = plt.stem(perc_non_zero,\
                use_line_collection=True)
                plt.setp(markers, marker='o', markersize=5, color='black',
                    markeredgecolor='darkorange', markeredgewidth=0)
                plt.setp(stemlines, color='darkorange', linewidth=0.5)
                plt.show()
            

            if self.feature_size is None:
                if sel_approach == "old":
            # Identify features that fulfill requirements set by user
                    sel_var = np.where(
                            (summary_df.iloc[4, :] >= cutoff_perc) & 
                            (summary_df.iloc[5, :] >= cutoff_means_ratio) & 
                            (summary_df.iloc[6, :] >= cutoff_mean_std_ratio\
                                        ))
                    sel_var_nzero = np.where(
                            (summary_df.iloc[4, :] >= cutoff_perc) & 
                            (summary_df.iloc[10, :] >= cutoff_means_ratio) & 
                            (summary_df.iloc[11, :] >= cutoff_mean_std_ratio\
                                        ))
                    
                elif sel_approach == "new":
                    sel_var = np.where(
                            (summary_df.iloc[4, :] >= cutoff_perc) & 
                            (summary_df.iloc[12, :] >= cutoff_means_ratio) & 
                            (summary_df.iloc[15, :] >= cutoff_mean_std_ratio\
                                        ))
                    sel_var_nzero = np.where(
                            (summary_df.iloc[4, :] >= cutoff_perc) & 
                            (summary_df.iloc[10, :] >= cutoff_means_ratio) & 
                            (summary_df.iloc[11, :] >= cutoff_mean_std_ratio\
                                        ))
                else:
                    print("sel_approach must be old or new")
                    

                select_feat_names = \
                [self.feat_names[ind] for ind in sel_var[0]]
            else:
                # if number of features selected is pre-set
                def normalize(vec):
                    if len(np.unique(np.array(vec)[~np.isnan(np.array(vec))])) == 1:
                        return vec
                    else:
                        return((vec-np.nanmin(vec))/(np.nanmax(vec)-np.nanmin(vec)))
                
                
                if sel_approach == "old":
                    scores_normalized = normalize(summary_df.iloc[4, :])
                    mean_frac_normalized = normalize(summary_df.iloc[5, :])
                    sd_frac_normalized = normalize(summary_df.iloc[6, :])
                elif sel_approach == "new":
                    scores_normalized = normalize(summary_df.iloc[4, :])
                    mean_frac_normalized = normalize(summary_df.iloc[12, :])
                    sd_frac_normalized = normalize(summary_df.iloc[15, :])
                else:
                    print("sel_approach must be old or new")
                
                fr = pd.Series(np.nansum(
                        pd.DataFrame([scores_normalized,\
                                      mean_frac_normalized, \
                                      sd_frac_normalized]), axis=0),\
                               index=summary_df.columns)
                select_feat_names = fr.nlargest(self.feature_size, 
                                                keep="all").index
                

            self.reduced_data = self.data[select_feat_names]
            self.spec_summary_df = summary_df
            
            if self.feature_size is None:
                select_feat_names_nzero = \
                [self.feat_names[ind] for ind in sel_var_nzero[0]]
                self.reduced_data_nzero = self.data[select_feat_names_nzero]
                return(summary_df, self.reduced_data, 
                       sel_var)  
            else:
                return(summary_df, self.reduced_data) 
    
    def get_spec_weights_dict(self, reg_param, l1_param):
        """
        Returns weights (coefficients) in a dictionary for models that were
        computed for a specifc regularisation paramter C. 
    
        
        INPUT
        ----
        reg_param: <int> Integer value for regularisation paramter C
        
        OUTPUT
        ------
        <dictionary>
            A dictionary holding weights of each computed model.
            Dictionary key: (l1_param, C, num_tt, num_w_init)
        """
        
        # Check if provided paramter was used in computations

        if reg_param not in self.reg_params and l1_param not in self.l1_params:
            print('No computation for this parameter combination available.')
            return None
        
        # Collect weights for requested regularisation parameter C
        else:
            self.spec_coef_dict = dict()
            for k in self.weight_dict.keys():
                if k[0] == l1_param and k[1] == reg_param:
                    self.spec_coef_dict[k] = self.weight_dict[k]
        
        return(self.spec_coef_dict)


    def get_spec_weights_df(self, reg_param, l1_param):
        """
        Returns weights (coefficients) of all computed models in a pandas
        dataframe for a specific regression parameter. 
          
        INPUT
        ----
        regression parameter
        
        OUTPUT
        ------
        <pandas dataframe>
            A dataframe holding weights of each computed model.
            Dictionary key: (l1_param, C, num_tt, num_w_init)
        """
        spec_weights_df = pd.DataFrame()
        for k in self.weight_dict.keys():
            if k[0] == l1_param and k[1] == reg_param:
                spec_weights_df = spec_weights_df.append( \
                        pd.DataFrame(self.weight_dict[k]))
        return(spec_weights_df)


    def get_spec_incorr_lables(self, reg_param, l1_param):
        """
        This method computes a summary of classifications across all models.
        Contains information on how often a sample has been mis-classfied.
        
        INPUT
        -----      
        reg_param: int, must be one of the regularisation parameters used in
                   computations
                
        OUTPUT
        ------        
        (DataFrame, list): DataFrame contains summary of misclassifications
                           list contains dataframes of predictions and ground
                           truths of each sample
        """
        
        # Check whether a regularisation parameter is given that computations
        # were done for
        if (reg_param is None) and (l1_param is None):

            for dataframe in self.pred_dict.keys():
                for count, tup in enumerate(zip(self.pred_dict[dataframe].y_test,\
                                                self.pred_dict[dataframe].y_pred)):
                    ind = self.pred_dict[dataframe].index[count]
                    
                    # Upgrade number of used as test object
                    self.incorrect_labels.loc[ind, "nr.test"] =\
                    (self.incorrect_labels.loc[ind, "nr.test"]) + 1
                    
                    if tup[0] != tup[1]:
                        # Upgrade number of wrongly classified
                        self.incorrect_labels.loc[ind, "nr.incorrect"] =\
                        (self.incorrect_labels.loc[ind, "nr.incorrect"]) + 1
            
            self.incorrect_labels["perc incorrect"] = \
            (self.incorrect_labels["nr.incorrect"]\
             / self.incorrect_labels["nr.test"]) * 100 
            return(self.incorrect_labels)
             
        else:

            # If regularisation parameter is given, compute summary. Get
            # prediction
            spec_pred_list = []
            
            for tt_split in range(self.num_tt):
                
                for w_init in range(self.num_w_init):
                    spec_pred_list.append(self.pred_dict[(l1_param,\
                                                          reg_param,\
                                                          tt_split,\
                                                          w_init)])
            #print(spec_pred_list)
        # Count number misclassifications for each object across all 
        # train-test splits            
            
            for dataframe in range(len(spec_pred_list)):
                for count, tup in enumerate(zip(spec_pred_list[dataframe].y_test,\
                                                spec_pred_list[dataframe].y_pred)):
                    ind = spec_pred_list[dataframe].index[count]
                    #print(ind)
                    # Upgrade number of used as test object
                    self.incorrect_labels.loc[ind, "nr.test"] =\
                    (self.incorrect_labels.loc[ind, "nr.test"]) + 1
                    
                    if tup[0] != tup[1]:
                        # Upgrade number of wrongly classified
                        self.incorrect_labels.loc[ind, "nr.incorrect"] =\
                        (self.incorrect_labels.loc[ind, "nr.incorrect"]) + 1
            
            self.incorrect_labels["perc incorrect"] = \
            (self.incorrect_labels["nr.incorrect"]\
             / self.incorrect_labels["nr.test"]) * 100 
            
            return(self.incorrect_labels, spec_pred_list)
            
            
    def get_indices(self):
        return self.indices
    
    def confusion_variance_plot(self, x_lab = "PC1", y_lab = "PC2"):
        """
        This method performs PCA on the reduced data and plots a seaborn 
        relplot.
        The plot is colored by true negative/positive and false 
        negative/positive 
        patiens and a neutral class which are in none of the "subsets"
        constructed by thresholding.
        ATTENTION: must run method get_spec_incorr_labes beforehand
        
        INPUT
        -----      
        x_lab: component on the x-axis
        y_lab: component on the y-axis
                
        OUTPUT
        ------        
        colored seaborn plot
        """
        block = self.reduced_data
        block = block.drop(axis=1,labels=(block.nunique())\
                           [block.nunique()==1].index)
        pca_model=ho.nipalsPCA(arrX=block.values, Xstand=True, cvType=None)
        block_scores = pd.DataFrame(pca_model.X_scores())
        block_scores.index = list(block.index)
        block_scores.columns = ['PC{0}'.format(x+1) for x in \
                                range(pca_model.X_scores().shape[1])]
        
        # create confusion matrix coloring
        label_matrix = self.incorrect_labels
        col = np.empty((np.shape(label_matrix)[0],), dtype=np.dtype('U100'))
        # neutral patients have an incorrectly labeled rate between 25 and 75 %
        # negative = 0
        # positive = 1
        col[:] = "neutral"
        col[np.where((label_matrix['perc incorrect']<=25) & \
                     (label_matrix['class'] == 0))[0]] = 'TN' 
        col[np.where((label_matrix['perc incorrect']<=25) & \
                     (label_matrix['class'] == 1))[0]] = 'TP'
        col[np.where((label_matrix['perc incorrect']>=75) & \
                     (label_matrix['class'] == 0))[0]] = 'FP'
        col[np.where((label_matrix['perc incorrect']>=75) & \
                     (label_matrix['class'] == 1))[0]] = 'FN'
        
        # add column to label_matrix and sort it to guarantee constant plot 
        # colors through different plots
        label_matrix['coloring'] = col
        block_scores = block_scores.merge(label_matrix.iloc[:,-1], \
                                          left_index=True, right_index =True)
        block_scores = block_scores.sort_values(by="coloring")
        self.xx = block_scores
        #sns relplot
        sns.relplot(x=x_lab,y=y_lab,data=block_scores\
                    ,hue=block_scores.iloc[:,-1])
        
# =============================================================================
#         Pred proba needs doesn't work. Problem with parallelization...
# =============================================================================
    def pred_proba_plot(self, reg_param, l1_param, patient_id,
                        binning="auto", lower=0, upper=1, kde=False, 
                        norm_hist=False):
        """
        This method produces histogram/density plots of the predicted 
        probabilities for patients.
        
        INPUT
        -----      
        reg_param: regression parameter
        patient_id: list of patients
        binning: binning procedure (auto, rice, sturges; see 
        https://www.answerminer.com/blog/binning-guide-ideal-histogram)
        ATTENTION: must run method pre_proba beforehand
                
        OUTPUT
        ------        
        seaborn plot
        """
        # different binning schemata
        # https://www.answerminer.com/blog/binning-guide-ideal-histogram
        for patient in patient_id:
            fig, ax = plt.subplots()
            data = self.rdict[reg_param, l1_param].loc[patient,:].dropna()

            if binning == "auto":
                bins = None
            if binning == "rice":
                bins = math.ceil(2*len(data)**(1./3.))
            if binning == "sturges":
                bins = math.ceil(math.log(len(data),2)) + 1
                
            ax=sns.distplot(data, axlabel ="ProbC1", 
                            bins=bins, 
                            color = 'darkblue',
                            hist_kws={'edgecolor':'darkblue'},
                            kde_kws={'linewidth': 3},
                            kde=kde,
                            norm_hist=norm_hist)
            ax.set(xlim=(lower, upper))
            ax.axvline(x=0.5, color='k', linestyle='--', label ="Threshold")
            ax.legend()
            if norm_hist == False:
                ax.set_ylabel('absolute frequencies')
            else:
                ax.set_ylabel('frequencies')
            ax.set_title('Patient: {0}, True class: {1}'.format(patient, \
                         self.target[patient]))
            
    def feasibility_study(self, test_data, test_labels, features, feature_size):
        # FS1
        pred_list0 = []
        for i in range(100):
            print(i)
            columns = np.random.RandomState(seed=i).choice(range(0,len(self.data.columns)),
                                    feature_size)
            sc = StandardScaler()
            traind = sc.fit_transform(self.data.iloc[:,columns])
            testd = sc.transform(test_data.iloc[:,columns])
            model = LogisticRegression(penalty='none', max_iter=8000, 
                                       solver="saga", random_state=0).\
                fit(traind,self.target)
            pred_list0.append(matthews_corrcoef(test_labels, \
                                                model.predict(testd)))
        print("Average score random feature drawing: ", np.mean(pred_list0))
        
        # FS2
        sc = StandardScaler()
        test_data.columns = self.data.columns
        pred_list1 = []
        
        traind = sc.fit_transform(self.data.loc[:,features])
        testd = sc.transform(test_data.loc[:,features])
        model = LogisticRegression(penalty='none', max_iter=8000, 
                                   solver="saga", random_state=0 ).\
                fit(traind,self.target)
        for i in range(100):
            pred_list1.append(matthews_corrcoef(
                    np.random.RandomState(seed=i).permutation(test_labels),\
                    model.predict(testd)))
        print("Average score permutation of test labels: ", np.mean(pred_list1))
        
       

       