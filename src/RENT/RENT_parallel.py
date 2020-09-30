# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:35:24 2020

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
from sklearn.metrics import f1_score, precision_score, recall_score, \
                            matthews_corrcoef, r2_score
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
    binary classification or regression dataset. Feature selection is done on
    multiple train test splits. The user can initiate interactions between 
    features that are included in the dataset and as such introduce 
    non-linearities.
    
    INPUT
    -----
    
    data: numpy array
    
    target: numpy array
    
    feat_names: list holding feature names
    
    scale: boolean, default=True
    
    C: list holding regularisation parameters for model
    
    l1_ratios: list holding ratios between l1 and l2 penalty
    
    poly: str, options: 'OFF', no feature interaction
                        'ON', (includes squared of features)
                        'ON_only_interactions', (only interactions, 
                                                 no squared features)
    
    testsize_range: tuple (low, high) range of random proportion of dataset to
    include in test set,
                    low and high are floats between 0 and 1, 
                    default (0.2, 0.6)
    
    scoring: str, options: 'accuracy', 'f1', 'precision', 'recall', 'matthews'
    
    method: str, options: 'logreg' for logistic regression
                          'linearSVC' for linear support vector classifier
                          'RM' for linear regression
    
    K: int, number of unique models in ensemble based on unique subsets of data

    
    OUTPUT
    ------
    None 
    """

    def __init__(self, data, target, feat_names=[],
                 scale=True, C=[1], poly='OFF',
                 testsize_range=(0.2, 0.6),
                 scoring='accuracy', method='logreg',
                 K=5, l1_ratios = [0.6],
                 verbose = 0):
        
        # Print parameters for checking
        print('Dim data:', np.shape(data))
        print('Dim target', np.shape(target))
        print('regularization parameters C:', C)
        print('l1_ratios:', l1_ratios)
        print('number of models in ensemble:', K)
        print('data type:', type(data))
        print('verbose:', verbose)


        # Define all objects needed later in methods below
        self.target = target
        self.C = C
        self.l1_ratios = l1_ratios
        self.K = K
        self.feat_names = feat_names
        self.scoring = scoring
        self.method = method
        self.testsize_range = testsize_range
        self.verbose = verbose
        
        
        # Check if data is dataframe and add index information
        if isinstance(data, pd.DataFrame):
            self.indices = data.index
        else:
            self.indices = list(range(data.shape[0]))


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
                
            flist = list(self.feat_names)
            flist.extend(polynom_feat_names)
            self.feat_names = flist
            self.data = pd.DataFrame(self.data)
            self.data.index = self.indices
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


    def run_parallel(self, tt_split):
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

        # Loop through all C 
        for C in self.C:
            # Loop through requested number of tt splits
            for l1 in self.l1_ratios:
                
                if self.method == "RM":
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
                    print('l1 = ', l1, 'C = ', C, ', TT split = ', tt_split)

                if self.method == 'logreg':
                    # Trian a logistic regreission model
                    model = LogisticRegression(solver='saga', 
                                            C=C,
                                            penalty='elasticnet',
                                            l1_ratio=l1,
                                            n_jobs=-1,
                                            max_iter=5000,
                                            random_state=0).\
                                            fit(X_train_std, y_train)
                    
                elif self.method == 'linSVC':
                    model = LinearSVC(penalty='l1',
                                    C=C,
                                    dual=False,
                                    max_iter=8000,
                                    random_state=0).\
                                    fit(X_train_std, y_train)
                
                elif self.method == "RM":
                    model = ElasticNet(alpha=1/C, l1_ratio=l1,
                                       max_iter=5000, random_state=0, \
                                       fit_intercept=False).\
                                       fit(X_train_std, y_train)

                # Get all weights (coefficients). Those that were selected
                # are non-zero, otherwise zero
                #print(logreg.coef_)
                mod_coef = model.coef_
                self.mcf = model.coef_
                if len(np.shape(mod_coef)) == 1:
                    mod_coef = mod_coef.reshape(1, len(mod_coef))
                self.weight_dict[(C, l1, tt_split)] = mod_coef
                self.weight_list.append(mod_coef)
                
                if self.method =="RM":
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
                    sf_inv = f1_score((1 - y_test), (1 - y_test_pred))
                    sm = matthews_corrcoef(y_test, y_test_pred)
                    sa = model.score(X_test_std, y_test)
                    
                    sc = [sa, sf, sf_inv, sm]
                    self.sc[(l1, C, tt_split)] = sc
                
                    sc_d = np.transpose(pd.DataFrame(sc))
                    sc_d.columns = ['acc', 'f1', 'f1 inv', 'matthews']
                    sc_d.index = [str(C)]
                    self.sc_pd =self.sc_pd.append(sc_d)
                
                #check if we need score_all and score_dict
                self.score_all[(C, l1, tt_split)] = score
                self.score_list_all.append(score)
                self.score_dict[(C, l1, tt_split)] = score
                self.score_list.append(score)
                
                # Collect true values and predictions in dictionary
                if(self.method != "RM" and self.method != "linSVC"):
                    res_df = pd.DataFrame({"y_test":y_test, \
                                           "y_pred": y_test_pred})
                    res_df.index = X_test.index
                
                    # self.pred_dict[(l1, C, tt_split)] = res_df
                    
                # calculate predict_proba for current train/test and weight
                # initialization
                    res_df = pd.DataFrame(
                            {"y_test":y_test, "y_pred": y_test_pred})
                    res_df.index=X_test.index
                    self.pred_dict[(C, l1, tt_split)] = res_df
                    self.p[(C, l1, tt_split)] = pd.DataFrame( \
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
            Dictionary key: (C, l1_ratios)
        """
        self.rdict = {}
        for C in self.C:
            for l1 in self.l1_ratios:
                count =  0
                vec = pd.DataFrame(np.nan, index= self.indices, \
                                   columns = ["remove"])
                for k in self.p.keys():
                   
                    if k[0] == C and k[1] == l1:
                        vec.loc[self.p[k].index,count] = \
                        self.p[k].iloc[:, 1].values
                        count = count+1
                        
                vec = vec.iloc[:, 1:]
                
                self.rdict[(C, l1)] = vec
        return self.rdict
    
    def run_analysis(self):
        """
        This method trains C * l1_ratio * num_tt models in total. The number
        of models using the same hyperparamter is num_tt.
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
                                                  self.K)
        
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
        # model. Kes are (l1_ratios, C, num_tt, num_w_init)
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
             map(delayed(self.run_parallel), range(self.K)))  
        ende = time.time()
        self.runtime = '{:5.3f}s'.format(ende-start)
            
    def get_scores_summary_by_regParam(self, showall=False):
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
            for l1 in self.l1_ratios:
                for C in self.C:
                    arr = np.empty((0, 4), int)
                    for k in self.sc.keys():
                        if k[0] == C and k[1] == l1:
                            arr = np.vstack((arr, self.sc[k]))
                            scores[(l1, C)] = np.transpose(pd.DataFrame(
                                    np.apply_along_axis(np.mean, 0, arr)))
                            scores[(l1, C)].columns =  \
                            ['acc', 'f1', 'f1 inv', 'matthews']
            return scores
        else:
            result_list=[]
            for l1 in self.l1_ratios:
                for C in self.C:
                    spec_result_list = []
                    for k in self.score_dict.keys():
                        if k[0] == C and k[1] == l1:
                            spec_result_list.append(self.score_dict[k])
                    result_list.append(spec_result_list)
            
        
            means=[]
            for r in range(len(result_list)):
                means.append(np.mean(result_list[r]))
    
            summary_df = pd.DataFrame(np.array(means).reshape(\
                                      len(self.l1_ratios), \
                                      len(self.C)), \
            index= self.l1_ratios, columns = self.C)
        
            return summary_df
    
    
    def get_average_zero_features(self):
        """
        Returns the average of features set to 0 for each pair (l1, C). 
        
        INPUT
        -----
        None
        
        OUTPUT
        ------
        <pandas DataFrame>: 
        """
        result_list = pd.DataFrame(index=self.l1_ratios,\
                                   columns=self.C)
        for l1 in self.l1_ratios:
            for C in self.C:
                count=0
                for tt_split in range(self.K):
        
                    nz = \
                    len(np.where(pd.DataFrame(self.weight_dict[(C, \
                                                                l1,\
                                                                tt_split)\
])==0)[0])
                    count = count + nz / len(self.feat_names)
                count = count / (self.K)
                result_list.loc[l1, C] = count
        return result_list

    
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
            Dictionary key: (C, num_tt)
        """
        
        if self.scoring == "all":
            print("Has to be adjusted for the scoring=all case")
        else:
            self.score_arr = np.vstack(self.score_list)
            scores_df = pd.DataFrame(self.score_arr)
            scores_df.index = list(self.weight_dict.keys())
            scores_df.columns = ['performance metric']
        
            return scores_df
    

    def get_spec_weights_summary(self, 
                                 C,
                                 l1_ratio,
                                 tau_1=0.9, 
                                 tau_2=0.9, 
                                 tau_3=1,
                                 plot=True):
        """
        Provides a summary of weights across all computed models for a 
        specific regularisation paramter chosen by user.
        
        INPUT
        -----
        C: <int> values for regularisation paramter
        
        tau_1: <int> or <float> 
            Cutoff critera for feature selection. Minimum Frequency of how 
            often a feature must have been selected across all models
            for given regularisation parameter. Choose value between 0 and 
            1 (in %).
        
        tau_2: <float>
            Cutoff criteria for feature selection. For a feature to be selected
            criteria 2 must be higher than cutoff_means_ratio.
        
        tau_3: <float>
            Cutoff criteria for feature selection. For a feature to be selected
            criteria 3 must be higher than cutoff_mean_std_ratio.
            
        
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


        if C not in self.C:
            print('No computation for this C parameter available.')
            return None
        elif l1_ratio not in self.l1_ratios:
            print('No computation for this l1 parameter available.')
            return None

        else:
            spec_weight_list = []
            
            # Loop through all train-test splits
            for tt_split in range(self.K):
                
                # Loop through all weight initialisations

                spec_weight_list.append(self.weight_dict[(C,\
                                                          l1_ratio,\
                                                          tt_split)])
            spec_weight_arr = np.vstack(spec_weight_list)  

            # Compute results based on weights
            counts_non_zero = np.count_nonzero(spec_weight_arr, axis=0)
            perc_non_zero = counts_non_zero / len(spec_weight_list) * 100
            
            # Statistics based on all weights
            means = np.mean(spec_weight_arr, axis=0)
            stds = np.std(spec_weight_arr, axis=0)

            # Statistics based on non-zero weights
            # mean_nzero, std_nzero, perc, mean_abs_nzero, element_count = \
            # np.apply_along_axis(self.zero_parameters, 0, spec_weight_arr)
            
            def non_zero_count_perc(vec):
                feature_count = len(np.where(vec != 0)[0])
                return feature_count/len(vec)
            
            perc = np.apply_along_axis(non_zero_count_perc, 0, spec_weight_arr)

            # signum function
            def sign_vote(x):
                return np.abs(np.sum(np.sign(x))) / len(x)
            sig_criterium = np.apply_along_axis(sign_vote,0,spec_weight_arr)
            
            # t-statistic
            t_test = t.cdf(
                    abs(means / np.sqrt((stds ** 2) / len(spec_weight_list))), \
           (len(spec_weight_list) - 1))
            
            # Construct a dataframe thaw holds the results compted above
            summary = np.vstack([perc,
                                 sig_criterium,
                                 t_test])
            
            summary_df = pd.DataFrame(summary)
            summary_df.index = ['perc non-zero',
                                'sig_criterium',
                                't_test']
            
            # Insert feature names in data frame
            summary_df.columns = self.feat_names
            
            # Make plot that shows percentages
            if plot == True:
                plt.figure(figsize=(10, 7))
                (markers, stemlines, baseline) = plt.stem(perc_non_zero,\
                use_line_collection=True)
                plt.setp(markers, marker='o', markersize=5, color='black',
                    markeredgecolor='darkorange', markeredgewidth=0)
                plt.setp(stemlines, color='darkorange', linewidth=0.5)
                plt.show()
            


            # Identify features that fulfill requirements set by use

            sel_var = np.where(
                    (summary_df.iloc[0, :] >= tau_1) & 
                    (summary_df.iloc[1, :] >= tau_2) & 
                    (summary_df.iloc[2, :] >= tau_3\
                                ))
                    

            select_feat_names = \
            [self.feat_names[ind] for ind in sel_var[0]]

            self.reduced_data = self.data[select_feat_names]
            
            summary_df.columns.name ='C={0}, l1={1}'.format(C, l1_ratio)
                
            return(summary_df, self.reduced_data, sel_var[0])  
    
    def get_spec_weights_dict(self, C, l1_ratio):
        """
        Returns weights (coefficients) in a dictionary for models that were
        computed for a specifc regularisation paramter C. 
    
        
        INPUT
        ----
        C: <int> Integer value for regularisation paramter C
        
        OUTPUT
        ------
        <dictionary>
            A dictionary holding weights of each computed model.
            Dictionary key: (l1_ratio, C, num_tt, num_w_init)
        """
        
        # Check if provided paramter was used in computations

        if C not in self.C and l1_ratio not in self.l1_ratios:
            print('No computation for this parameter combination available.')
            return None
        
        # Collect weights for requested regularisation parameter C
        else:
            self.spec_coef_dict = dict()
            for k in self.weight_dict.keys():
                if k[0] == C and k[1] == l1_ratio:
                    self.spec_coef_dict[k] = self.weight_dict[k]
        
        return(self.spec_coef_dict)


    def get_spec_weights_df(self, C, l1_ratio):
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
            Dictionary key: (l1_ratio, C, num_tt, num_w_init)
        """
        spec_weights_df = pd.DataFrame()
        for k in self.weight_dict.keys():
            if k[0] == C and k[1] == l1_ratio:
                spec_weights_df = spec_weights_df.append( \
                        pd.DataFrame(self.weight_dict[k]))
        return(spec_weights_df)


    def get_spec_incorr_lables(self, C, l1_ratio):
        """
        This method computes a summary of classifications across all models.
        Contains information on how often a sample has been mis-classfied.
        
        INPUT
        -----      
        C: int, must be one of the regularisation parameters used in
                   computations
                
        OUTPUT
        ------        
        (DataFrame, list): DataFrame contains summary of misclassifications
                           list contains dataframes of predictions and ground
                           truths of each sample
        """
        
        # Check whether a regularisation parameter is given that computations
        # were done for
        if (C is None) and (l1_ratio is None):

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
            
            for tt_split in range(self.K):

                spec_pred_list.append(self.pred_dict[(C,\
                                                      l1_ratio,\
                                                      tt_split)])
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
    
    def confusion_variance_plot(self, x_lab="PC1", y_lab="PC2"):
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
        block = block.drop(axis=1, labels=(block.nunique())\
                           [block.nunique()==1].index)
        pca_model=ho.nipalsPCA(arrX=block.values, Xstand=True, cvType=None)
        block_scores = pd.DataFrame(pca_model.X_scores())
        block_scores.index = list(block.index)
        block_scores.columns = ['PC{0}'.format(x+1) for x in \
                                range(pca_model.X_scores().shape[1])]
        
        # create confusion matrix coloring
        label_matrix = self.incorrect_labels
        col = np.empty((np.shape(label_matrix)[0],), dtype=np.dtype('U100'))
        
        # neutral objects have an incorrectly labeled rate between 25 and 75 %
        # negative = 0
        # positive = 1
        col[:] = "neutral"
        col[np.where((label_matrix['perc incorrect'] <= 25) & \
                     (label_matrix['class'] == 0))[0]] = 'TN' 
        col[np.where((label_matrix['perc incorrect'] <= 25) & \
                     (label_matrix['class'] == 1))[0]] = 'TP'
        col[np.where((label_matrix['perc incorrect'] >= 75) & \
                     (label_matrix['class'] == 0))[0]] = 'FP'
        col[np.where((label_matrix['perc incorrect'] >= 75) & \
                     (label_matrix['class'] == 1))[0]] = 'FN'
        
        # add column to label_matrix and sort it to guarantee constant plot 
        # colors through different plots
        label_matrix['coloring'] = col
        block_scores = block_scores.merge(label_matrix.iloc[:, -1], \
                                          left_index=True, right_index=True)
        block_scores = block_scores.sort_values(by="coloring")
        self.xx = block_scores
        #sns relplot
        sns.relplot(x=x_lab, y=y_lab, data=block_scores\
                    ,hue=block_scores.iloc[:, -1])
        

    def pred_proba_plot(self, C, l1_ratio, object_id,
                        binning="auto", lower=0, upper=1, kde=False, 
                        norm_hist=False):
        """
        This method produces histogram/density plots of the predicted 
        probabilities for objects.
        
        INPUT
        -----      
        C: regression parameter
        object_id: list of objects
        binning: binning procedure (auto, rice, sturges; see 
        https://www.answerminer.com/blog/binning-guide-ideal-histogram)
        ATTENTION: must run method pre_proba beforehand
                
        OUTPUT
        ------        
        seaborn plot
        """
        # different binning schemata
        # https://www.answerminer.com/blog/binning-guide-ideal-histogram
        for obj in object_id:
            fig, ax = plt.subplots()
            data = self.rdict[C, l1_ratio].loc[obj,:].dropna()

            if binning == "auto":
                bins = None
            if binning == "rice":
                bins = math.ceil(2 * len(data) ** (1. / 3.))
            if binning == "sturges":
                bins = math.ceil(math.log(len(data), 2)) + 1
            sns.set(font_scale=0.5)
            sns.set_style("white")
            ax=sns.distplot(data, 
                            bins=bins, 
                            color = 'darkblue',
                            hist_kws={'edgecolor':'darkblue'},
                            kde_kws={'linewidth': 3},
                            kde=kde,
                            norm_hist=norm_hist)
            ax.set(xlim=(lower, upper))
            ax.axvline(x=0.5, color='k', linestyle='--', label ="Threshold")
            ax.legend(fontsize=10)
            
            if norm_hist == False:
                ax.set_ylabel('absolute frequencies', fontsize=10)
                ax.set_xlabel('ProbC1', fontsize=10)
            else:
                ax.set_ylabel('frequencies')
                ax.set_xlabel('ProbC1')
            ax.set_title('Object: {0}, True class: {1}'.format(obj, \
                         self.target[obj]), fontsize=10)
            
    def feasibility_study(self, test_data, test_labels, features, feature_size):
        # FS1
        pred_list0 = []
        for i in range(100):
            columns = np.random.RandomState(seed=i).choice(
                range(0,len(self.data.columns)),
                                    feature_size)
            sc = StandardScaler()
            traind = sc.fit_transform(self.data.iloc[:, columns])
            testd = sc.transform(test_data.iloc[:, columns])
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
        
        traind = sc.fit_transform(self.data.loc[:, features])
        testd = sc.transform(test_data.loc[:, features])
        model = LogisticRegression(penalty='none', max_iter=8000, 
                                   solver="saga", random_state=0 ).\
                fit(traind,self.target)
        
        for i in range(100):
            pred_list1.append(matthews_corrcoef(
                    np.random.RandomState(seed=i).permutation(test_labels),\
                    model.predict(testd)))
        print("Average score permutation of test labels: ", np.mean(pred_list1))
        
       

       