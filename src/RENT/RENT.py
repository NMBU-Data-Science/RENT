# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:28:51 2020

@author: ajenul
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import sys
import time
import warnings
import hoggorm as ho
import hoggormplot as hopl

from abc import ABC, abstractmethod
from itertools import combinations, combinations_with_replacement
from joblib import Parallel, delayed



from sklearn.linear_model import LogisticRegression, ElasticNet, \
    LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, \
                            matthews_corrcoef, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC

from scipy.stats import t


class RENT_Base(ABC):
    """
    This is the base class for RENT_Classification and RENT_Regression.
    """
    
    @abstractmethod
    def feasibility_study(self, test_data, test_labels, K_feas):
        pass
    
    @abstractmethod
    def run_parallel(self, K):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def parameter_selection(self, C_params, l1_params, n_splits, testsize_range):
        pass
    
    @abstractmethod
    def summary_objects(self):
        pass
    
    def selectFeatures(self, tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975):
        """
        Selectes features based on the cutoff values for tau_1_cutoff, tau_2_cutoff and tau_3_cutoff.
        
        Parameters
        ----------
        tau_1_cutoff : <float>
            Cutoff value for tau_1 criterion. Choose value between 0 and 
            1. The default is 0.9.
        tau_2_cutoff : <float>
            Cutoff value for tau_2 criterion. Choose value between 0 and 
            1. The default is 0.9.
        tau_3_cutoff : <float>
            Cutoff value for tau_3 criterion. Choose value between 0 and 
            1. The default is 0.975.

        Returns
        -------
        Numpy array holduing indices of selected features.

        """
        weight_list = []
        #Loop through all K models
        for K in range(self.K):
            weight_list.append(self.weight_dict[(self._best_C,
                                                 self._best_l1_ratio,
                                                 K)])
        weight_array = np.vstack(weight_list)
        
        #Compute results based on weights
        counts = np.count_nonzero(weight_array, axis=0)
        self._perc = counts / len(weight_list)
        means = np.mean(weight_array, axis=0)
        stds = np.std(weight_array, axis=0)
        signum = np.apply_along_axis(self.sign_vote, 0, weight_array)
        t_test = t.cdf(
            abs(means / np.sqrt((stds ** 2) / len(weight_list))), \
                (len(weight_list)-1))
            
        # Conduct a dataframe that stores the results for the criteria
        summary = np.vstack([self._perc, signum, t_test])
        self.summary_df = pd.DataFrame(summary)
        self.summary_df.index = ['tau_1', 'tau_2', 'tau_3']
        self.summary_df.columns = self.feat_names
        
        self.sel_var = np.where(
                (self.summary_df.iloc[0, :] >= tau_1_cutoff) & 
                (self.summary_df.iloc[1, :] >= tau_2_cutoff) & 
                (self.summary_df.iloc[2, :] >= tau_3_cutoff\
                            ))[0]
        return self.sel_var
    
    def summary_criteria(self):
        """
        Returns
        -------
        Summary statistic of the selection criteria tau_1, tau_2 and tau_3
        for each feature. Also prints out summary statistic.
        """
        return self.summary_df
        
    def plot_selection_frequency(self):
        """
        Plots tau_1 for each feature. 

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 7))
        (markers, stemlines, baseline) = plt.stem(self._perc,\
        use_line_collection=True)
        plt.setp(markers, marker='o', markersize=5, color='black',
            markeredgecolor='darkorange', markeredgewidth=0)
        plt.setp(stemlines, color='darkorange', linewidth=0.5)
        plt.show()

    
    def get_weight_distributions(self):
        """
        Feature weights over the K models (Beta matrix in paper).

        Returns
        -------
        A data frame holding weight distribution for each feature. 
        Weights were collected across K models in the ensemble.
        
        rows: represents weights of one model
        columns: represents weights across models in ensemble for each feature

        """
        weights_df = pd.DataFrame()
        for k in self.weight_dict.keys():
            if k[0] == self._best_C and k[1] == self._best_l1_ratio:
                weights_df = weights_df.append( \
                        pd.DataFrame(self.weight_dict[k]))
        weights_df.index = list(range(self.K))
        return(weights_df)
    
        
    def plot_object_PCA(self, cl=0):
        """
        Applies principal component analysis on data containing only selected features.

        For classification:
            - user may select from the following:
                - PCA on class 0
                - PCA on class 1 
                - PCA on both classes
            - colouring of PCA scores depends on number of misclassfications across
              ensemble predictions.
        
        For regression:
        PCA applied to all samples. Colouring by average absolute error across
        ensemble predictions.

        Parameters
        ----------
        cl : <int> or <str>
            - For classification problem: 
                - <int>: 0 or 1 for class 0 or class 1, respectively; 
                - <str>: 'both' for both classes.
            
            - For regression problem:
                - <str>: 'continuous'


        Returns
        -------
        None.

        """
        if cl != 'continuous':
            dat = pd.merge(self.data, self.incorrect_labels.iloc[:,[1,-1]], \
                                 left_index=True, right_index=True)
            variables = list(self.sel_var)
            variables.extend([-2,-1])
        else:
            dat = pd.merge(self.data, self.incorrect_labels.iloc[:,-1], \
                                     left_index=True, right_index=True)
            variables = list(self.sel_var)
            variables.extend([-1])

        if cl == 'both' or cl == 'continuous':
            data = dat.iloc[:,variables]
        else:
            data = dat.iloc[np.where(dat.iloc[:,-2]==cl)[0],variables]
            
        if cl != 'continuous':
            data = data.sort_values(by='% incorrect')

            pca_model = ho.nipalsPCA(arrX=data.iloc[:,:-2].values, \
                                       Xstand=True, cvType=['loo'])

        else:
            pca_model = ho.nipalsPCA(arrX=data.iloc[:,:-1].values, \
                                       Xstand=True, cvType=['loo'])
                
                
        scores = pd.DataFrame(pca_model.X_scores())
        scores.index = list(data.index)
        scores.columns = ['PC{0}'.format(x+1) for x in \
                                 range(pca_model.X_scores().shape[1])]
        scores['coloring'] = data.iloc[:,-1]

        fig, ax = plt.subplots()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Scatterplot')
        ax.set_facecolor('silver')
        
        # plot
        if cl == 0:
            plt.scatter(scores['PC1'], scores['PC2'], c= scores['coloring'], 
                        cmap='Greens')
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 0')
        elif cl == 1:
            plt.scatter(scores['PC1'], scores['PC2'], c= scores['coloring'], 
                        cmap='Reds')
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 1')
        elif cl == 'both':
            zeroes = np.where(data.iloc[:,-2]==0)[0]
            ones = np.where(data.iloc[:,-2]==1)[0]

            plt.scatter(scores.iloc[zeroes,0], 
                        scores.iloc[zeroes,1], 
                        c= scores.iloc[zeroes,-1], 
                        cmap='Greens',
                        marker='*',
                        alpha=0.5)
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 0')
            plt.scatter(scores.iloc[ones,0], 
                        scores.iloc[ones,1], 
                        c= scores.iloc[ones,-1], 
                        cmap='Reds',
                        alpha=0.5)
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 1')
            
            
            mlist = []
            col_list = []

            for i in range(len(data.index)):
                if data.iloc[i,-2]==0:
                    mlist.append("*")
                else:
                    mlist.append("o")
        

            for i in range(len(data.index)):
                if data.iloc[i,-2]==0 and data.iloc[i,-1]==0:
                    col_list.append('honeydew')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]==0:
                    col_list.append('snow')
                elif data.iloc[i,-2]==0 and data.iloc[i,-1]>0 and data.iloc[i,-1]<50:
                    col_list.append('mediumspringgreen')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]>0 and data.iloc[i,-1]<50:
                    col_list.append('tomato')
                    col_list.append('green')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]>=50 and data.iloc[i,-1]<100: 
                    col_list.append('red')   
                elif data.iloc[i,-2]==0 and data.iloc[i,-1]==100:
                    col_list.append('darkgreen')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]==100: 
                    col_list.append('maroon')
                else:
                    col_list.append(np.nan)

            for i in range(len(mlist)):
                plt.scatter(scores.iloc[i,0], scores.iloc[i,1],
                            marker=mlist[i], c=col_list[i])
                            
        elif cl == 'continuous':
            plt.scatter(scores.iloc[:,0], scores.iloc[:,1], 
                        c=scores.iloc[:,-1],
                        cmap='YlOrRd')
            cbar = plt.colorbar()
            cbar.set_label('average absolute error')

        objnames = list(data.index.astype('str'))
        hopl.plot(pca_model, plots=[1,2,3,4,6], 
                  objNames=objnames, XvarNames=list(data.columns[:-2]))
        
        
    def get_enetParam_matrices(self):
        """
        Returns
        -------
        Returns three pandas data frames showing result for all combinations
        of l1_ratio and C.

        dataFrame_1: holds average scores for predictive performance
        dataFrame_2: holds average percentage of how many feature weights were set to zero
        dataFrame_3: holds harmonic means based from values of dataFrame_1 and dataFrame_2
        """
        # return scores, zeroes,.. matrices
        return self._scores_df, self._zeroes_df, self._combination
    
    def get_enet_params(self):
        """
        Returns
        -------
        A tuple holding (C, l1_ratio) for the best average predictive performance. This
        combination of C l1_ratio will be used in subsequent class methods.
        """
        return self._best_C, self._best_l1_ratio
    
    def set_enet_params(self, C, l1):
        self._best_C = C
        self._best_l1_ratio = l1
        
    def inv(self, num):
        #invert a number except 0
        if num == 0:
            return np.inf
        elif num == np.inf:
            return 0
        else:
            return num ** -1

    def sign_vote(self, arr):
        return np.abs(np.sum(np.sign(arr))) / len(arr)
    
    def get_runtime(self):
        return self._runtime



class RENT_Classification(RENT_Base):
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
    
    classifier: str, options: 'logreg' for logistic regression
                       'linearSVC' for linear support vector classifier
    
    K: int, number of unique train-test splits
    
    verbose: print something
    
    OUTPUT
    ------
    None 
    """

    def __init__(self, data, target, feat_names=[], C=[1,10], l1_ratios = [0.6],
                 parameter_selection=True, poly='OFF', 
                 testsize_range=(0.2, 0.6), scoring='accuracy', 
                 method='logreg', K=5, verbose = 0):
        
        # Print parameters for checking
        print('data dimension:', np.shape(data), ' data type:', type(data))
        print('target dimension', np.shape(target))
        print('regularization parameters C:', C)
        print('elastic net l1_ratios:', l1_ratios)
        print('number of models in ensemble:', K)
        print('classification method:', method)
        print('verbose:', verbose)


        # Define all objects needed later in methods below
        self.target = target
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
            print('No feature names found - automatic generate feature names.')
            
            for ind in range(1, np.shape(data)[1] + 1):
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
            sys.exit('Value for paramter "poly" not regcognised.')
        
        if parameter_selection == True:
            self.C, self.l1_ratios = self.parameter_selection(C_params=C, 
                                                              l1_params=l1_ratios)
            self.C = [self.C]
            self.l1_ratios = [self.l1_ratios]
        else:
            self.C = C
            self.l1_ratios = l1_ratios
            
    def run_parallel(self, K):
        """
        Parallel computation of for loops. Parallelizes the number of models (K)
        as this is the parameter with most varying values.
        
        INPUT
        -----
        K: range of train-test splits
        
        OUTPUT
        ------
        None 
        """

        # Loop through all C 
        for C in self.C:
            # Loop through requested number of tt splits
            for l1 in self.l1_ratios:
                
                X_train, X_test, y_train, y_test = train_test_split(
                      self.data, self.target, 
                      test_size=self.random_testsizes[K],
                      stratify=self.target, random_state=K)
                
#                self.train_sets.append(X_train)
                self.X_test = X_test
                
                # Initialise standard scaler and compute mean and STD from 
                # training data. 
                # Transform train and test dataset
                sc = StandardScaler()
                sc.fit(X_train)
                X_train_std = sc.transform(X_train)
                X_test_std = sc.transform(X_test)
                
                if self.verbose > 0:
                    print('C = ', C, 'l1 = ', l1, ', TT split = ', K)

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
                else:
                    sys.exit('No valid classification method.')
                    
                # Get all weights (coefficients). Those that were selected
                # are non-zero, otherwise zero
                #print(logreg.coef_)
                self.weight_dict[(C, l1, K)] = model.coef_
                self.weight_list.append(model.coef_)
                
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
                elif self.scoring == 'mcc':
                    y_test_pred = model.predict(X_test_std)
                    score = matthews_corrcoef(y_test, y_test_pred)
                
                #check if we need score_all and score_dict
                self.score_dict[(C, l1, K)] = score
                self.score_list.append(score)
                
                # Collect true values and predictions in dictionary
               
                predictions = pd.DataFrame({'y_test':y_test, \
                                       'y_pred': y_test_pred})
                predictions.index = X_test.index

                # calculate predict_proba for current train/test and weight
                # initialization
                self.predictions_dict[(C, l1, K)] = predictions
                if(self.method == 'logreg'):
                    self.probas[(C, l1, K)] = pd.DataFrame( \
                           model.predict_proba(X_test_std), index = \
                                X_test.index)

    def train(self):
        """
        This method trains C * l1_ratio * K models in total. The number
        of models using the same hyperparamter is K.
        For each model elastic net regularisation is applied for variable 
        selection.
         
        
        INPUT
        -----
        None
        
        OUTPUT
        ------
        None 
        """
        #check if this is needed and does what it should (probably doesn't work because of parallelization)
        np.random.seed(0)
        self.random_testsizes = np.random.uniform(self.testsize_range[0],
                                                  self.testsize_range[1], 
                                                  self.K)
        
        # Initiate a dictionary holding coefficients for each model. Keys are
        # (C, K, num_w_init)
        self.weight_dict = {}
        
        # Initiate a dictionary holding computed performance metric for each
        self.score_dict = {}


        # Collect all coefs in a list that will be converted to a numpy array
        self.weight_list = []
        
        # Collect all coefs in a list that will be converted to a numpy array
        self.score_list = []
        
        # Initialize a dictionary to predict incorrect labels
        self.predictions_dict = {}
        
        # store
        self.probas = {}
        self.pred_proba_dict = {}

        # stop runtime
        start = time.time()
        # Call parallelization function 
        Parallel(n_jobs=-1, verbose=0, backend='threading')(
             map(delayed(self.run_parallel), range(self.K)))  
        ende = time.time()
        self._runtime = '{:5.3f}s'.format(ende-start)
        
        # Build a dictionary with the prediction probabilities
        for C in self.C:
            for l1 in self.l1_ratios:
                count =  0
                vec = pd.DataFrame(np.nan, index= self.indices, \
                                   columns = ['remove'])
                for k in self.probas.keys():
                   
                    if k[0] == C and k[1] == l1:
                        vec.loc[self.probas[k].index,count] = \
                        self.probas[k].iloc[:, 1].values
                        count += 1
                        
                vec = vec.iloc[:, 1:]
                
                self.pred_proba_dict[(C, l1)] = vec
                
                
        # find best parameter setting and matrices
        result_list=[]
        for l1 in self.l1_ratios:
            for C in self.C:
                spec_result_list = []
                for k in self.score_dict.keys():
                    if k[0] == C and k[1] ==l1:
                        spec_result_list.append(self.score_dict[k])
                result_list.append(spec_result_list)
    
        means=[]
        for r in range(len(result_list)):
            means.append(np.mean(result_list[r]))

        self._scores_df = pd.DataFrame(np.array(means).reshape(\
                                  len(self.l1_ratios), \
                                  len(self.C)), \
        index= self.l1_ratios, columns = self.C)
        
        self._zeroes_df = pd.DataFrame(index = self.l1_ratios,\
                                   columns=self.C)
        for l1 in self.l1_ratios:
            for C in self.C:
                count = 0
                for K in range(self.K):
                    nz = \
                    len(np.where(pd.DataFrame(self.weight_dict[(C, l1, K)\
])==0)[0])
                    count = count + nz / len(self.feat_names)
                count = count / (self.K)
                self._zeroes_df.loc[l1, C] = count
                
        if len(self.C)>1 or len(self.l1_ratios)>1:
            normed_scores = pd.DataFrame((self._scores_df-np.nanmin(
                self._scores_df.values))/ \
                (np.nanmax(self._scores_df.values)-np.nanmin(
                    self._scores_df.values)))
            normed_zeroes = pd.DataFrame((self._zeroes_df-np.nanmin(
                self._zeroes_df.values))/ \
                (np.nanmax(self._zeroes_df.values)-np.nanmin(
                    self._zeroes_df.values)))
            normed_zeroes = normed_zeroes.astype('float')
    
            self._combination = (normed_scores.copy().applymap(self.inv) + \
                                        normed_zeroes.copy().applymap(
                                            self.inv)).applymap(self.inv)
        else:
            self._combination = (self._scores_df.copy().applymap(self.inv) + \
                                 self._zeroes_df.copy().applymap(
                                     self.inv)).applymap(self.inv)
        
        self._scores_df.columns.name = 'Scores'
        self._zeroes_df.columns.name = 'Zeroes'
        self._combination.columns.name = 'Harmonic Mean'
        
        best_row, best_col  = np.where(
            self._combination == np.nanmax(self._combination.values))
        self._best_l1_ratio = self._combination.index[np.nanmax(best_row)]
        self._best_C = self._combination.columns[np.nanmin(best_col)]
        
    def parameter_selection(self, 
                        C_params, 
                        l1_params, 
                        n_splits=5, 
                        testsize_range=(0.25,0.25)):
        """
        

        Parameters
        ----------
        C_params : TYPE
            DESCRIPTION.
        l1_params : TYPE
            DESCRIPTION.
        n_splits : TYPE, optional
            DESCRIPTION. The default is 5.
        testsize_range : TYPE, optional
            DESCRIPTION. The default is (0.25,0.25).

        Returns
        -------
        None.

        """
            
        # if len(np.shape(labels)) == 2:
        #     labels = labels.squeeze()
        
        skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
        
        scores_df = pd.DataFrame(np.zeros, index=l1_params, columns=C_params)
        zeroes_df = pd.DataFrame(np.zeros, index=l1_params, columns=C_params)
        
        
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
            for reg in C_params:
                scores = list()
                zeroes = list()
                for train, test in skf.split(self.data, self.target):
                    
                    sc = StandardScaler()
                    train_data = sc.fit_transform(self.data.iloc[train, :])
                    train_target = self.target[train]
                    test_data_split = sc.transform(self.data.iloc[test, :])
                    test_target = self.target[test]
                    sgd = LogisticRegression(penalty="elasticnet", C=reg, 
                                             solver="saga", l1_ratio=l1, 
                                             random_state=0)
                    sgd.fit(train_data, train_target)
                    
                    params = np.where(sgd.coef_ != 0)[1]
        
                    
                    if len(params) == 0:
                        scores.append(np.nan)
                        zeroes.append(np.nan)
                    else:
                        zeroes.append((len(self.data.columns)-len(params))\
                                      /len(self.data.columns))
                        
                        sc = StandardScaler()
                        train_data_1 = sc.fit_transform(train_data[:,params])
                        test_data_1 = sc.transform(test_data_split[:, params])
                        model = LogisticRegression(penalty='none', 
                                                   max_iter=8000, 
                                                   solver="saga",
                                                   random_state=0).\
                                fit(train_data_1, train_target)
                        scores.append(matthews_corrcoef(test_target, \
                                        model.predict(test_data_1)))
                                
                scores_df.loc[l1, reg] = np.nanmean(scores)
                zeroes_df.loc[l1, reg] = np.nanmean(zeroes)
    
        
        Parallel(n_jobs=-1, verbose=0, backend="threading")(
             map(delayed(run_parallel), l1_params))  
    
        normed_scores = (scores_df-np.nanmin(scores_df.values))\
        /(np.nanmax(scores_df.values)-np.nanmin(scores_df.values))
        normed_zeroes = (zeroes_df-np.nanmin(zeroes_df.values))\
        /(np.nanmax(zeroes_df.values)-np.nanmin(zeroes_df.values))
        
        combination = (normed_scores.copy().applymap(self.inv) + \
                       normed_zeroes.copy().applymap(self.inv)
                       ).applymap(self.inv)
        best_combination_row, best_combination_col = np.where(combination == \
                                                  np.nanmax(combination.values))
        best_l1 = combination.index[np.nanmax(best_combination_row)]
        best_C = combination.columns[np.nanmin(best_combination_col)]
        
        return(best_C, best_l1)
                
        
    def summary_objects(self):
        # incorrect probabilities dataframe
        """
        This method computes a summary of classifications across all models.
        Contains information on how often a sample has been mis-classfied.
        """
        
        #check that train has been run before

        self.incorrect_labels = pd.DataFrame({'# test':np.repeat\
                                      (0, np.shape(self.data)[0]),
                                      'class':self.target,
                                      '# incorrect':np.repeat\
                                      (0, np.shape(self.data)[0])})
        self.incorrect_labels.index=self.indices.copy()
        
        specific_predictions = []
        for K in range(self.K):
            specific_predictions.append(
                self.predictions_dict[(self._best_C, self._best_l1_ratio, K)])
        for dataframe in range(len(specific_predictions)):
            for count, tup in enumerate(
                    zip(specific_predictions[dataframe].y_test, \
                        specific_predictions[dataframe].y_pred)):
                ind = specific_predictions[dataframe].index[count]
                
                # Upgrade ind by one if used as test object
                self.incorrect_labels.loc[ind,'# test'] += 1
                if tup[0] != tup[1]:
                    # Upgrade number of incorrectly classified
                    self.incorrect_labels.loc[ind,'# incorrect'] += 1
        
        self.incorrect_labels['% incorrect'] = \
        (self.incorrect_labels["# incorrect"] \
             / self.incorrect_labels['# test']) * 100 
        
        return self.incorrect_labels
    
    
    def get_object_probabilities(self):
        # predicted probabilities only if Logreg
        if self.method != 'logreg':
            return warnings.warn('Classification method must be "logreg"!')
        else:
            self.pp_data = self.pred_proba_dict[
                (self._best_C, self._best_l1_ratio)].copy()
            
            self.pp_data.columns = ['K({0})'.format(x+1) \
                                        for x in range(
                                                self.pp_data.shape[1])]
            return self.pp_data
        
    def plot_object_probabilities(self, object_id, binning='auto', lower=0,
                                  upper=1, kde=False, norm_hist=False):
        """
        Histograms of predicted probabilities. Check scalings etc.

        Parameters
        ----------
        object_id : TYPE
            DESCRIPTION.
        binning : TYPE, optional
            DESCRIPTION. The default is 'auto'.
        lower : TYPE, optional
            DESCRIPTION. The default is 0.
        upper : TYPE, optional
            DESCRIPTION. The default is 1.
        kde : TYPE, optional
            DESCRIPTION. The default is False.
        norm_hist : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        # different binning schemata
        # https://www.answerminer.com/blog/binning-guide-ideal-histogram
        target_objects = pd.DataFrame(self.target)
        target_objects.index = self.pred_proba_dict[self._best_C, \
                              self._best_l1_ratio].index
        self.t = target_objects
        for obj in object_id:
            fig, ax = plt.subplots()
            data = self.pred_proba_dict[self._best_C, \
                              self._best_l1_ratio].loc[obj,:].dropna()

            if binning == "auto":
                bins = None
            if binning == "rice":
                bins = math.ceil(2 * len(data) ** (1./3.))
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
                         target_objects.loc[obj,0]), fontsize=10)
                


    def feasibility_study(self, test_data, test_labels, K_feas, metric='mcc',
                          alpha=0.05):
        """
        Feasibiliyt study as in paper. p-value has to be added.

        Parameters
        ----------
        test_data : TYPE
            DESCRIPTION.
        test_labels K_feas : TYPE
            DESCRIPTION.
        metric : TYPE, optional
            DESCRIPTION. The default is 'mcc'.

        Returns
        -------
        None.

        """
        
        
        # RENT prediction
        sc = StandardScaler()
        train_RENT = sc.fit_transform(self.data.iloc[:, self.sel_var])
        test_RENT = sc.transform(test_data.iloc[:, self.sel_var])
        if self.method == 'logreg':
                model = LogisticRegression(penalty='none', max_iter=8000, 
                                           solver="saga", random_state=0).\
                    fit(train_RENT,self.target)
        else:
            print("something")
            
        if metric == 'mcc':
                score = matthews_corrcoef(test_labels, model.predict(test_RENT))
        elif metric == 'f1':
            score = f1_score(test_labels, model.predict(test_RENT))
        elif metric == 'acc':
            score = accuracy_score(test_labels, model.predict(test_RENT))
        
        
        # FS1
        FS1 = []
        for K in range(K_feas):
            # Randomly select features (# features = # RENT features selected)
            columns = np.random.RandomState(seed=K).choice(
                range(0,len(self.data.columns)),
                                    len(self.sel_var))
            sc = StandardScaler()
            train_FS1 = sc.fit_transform(self.data.iloc[:, columns])
            test_FS1 = sc.transform(test_data.iloc[:, columns])
            if self.method == 'logreg':
                model = LogisticRegression(penalty='none', max_iter=8000, 
                                           solver="saga", random_state=0).\
                    fit(train_FS1,self.target)
            else:
                print("something")
            if metric == 'mcc':
                FS1.append(matthews_corrcoef(test_labels, \
                                                    model.predict(test_FS1)))
            elif metric == 'f1':
                FS1.append(f1_score(test_labels, \
                                                    model.predict(test_FS1)))
            elif metric == 'acc':
                FS1.append(accuracy_score(test_labels, \
                                                    model.predict(test_FS1)))
                    
        p_value_FS1 = sum(FS1 > score) / len(FS1)
        print("p-value average score random feature drawing: ", p_value_FS1)
        if p_value_FS1 <= alpha:
            print('With a significancelevel of ', alpha, ' H0 is rejected.')
        else:
            print('With a significancelevel of ', alpha, ' H0 is accepted.')
            
            
        # FS2
        sc = StandardScaler()
        test_data.columns = self.data.columns
        FS2 = []
        
        train_FS2 = sc.fit_transform(self.data.iloc[:,self.sel_var])
        test_FS2 = sc.transform(test_data.iloc[:, self.sel_var])
        if self.method == 'logreg':
            model = LogisticRegression(penalty='none', max_iter=8000, 
                                       solver="saga", random_state=0 ).\
                    fit(train_FS2, self.target)
        else:
            print("add model")
            
        for K in range(K_feas):
            if metric == 'mcc':
                FS2.append(matthews_corrcoef(
                        np.random.RandomState(seed=K).permutation(test_labels),\
                        model.predict(test_FS2)))
            elif metric == 'f1':
                FS2.append(f1_score(
                        np.random.RandomState(seed=K).permutation(test_labels),\
                        model.predict(test_FS2)))
            elif metric == 'acc':
                FS2.append(accuracy_score(
                        np.random.RandomState(seed=K).permutation(test_labels),\
                        model.predict(test_FS2)))

        p_value_FS2 = sum(FS2 > score) / len(FS2)
        print("p-value score permutation of test labels: ", p_value_FS2)
        if p_value_FS2 <= alpha:
            print('With a significancelevel of ', alpha, ' H0 is rejected.')
        else:
            print('With a significancelevel of ', alpha, ' H0 is accepted.')

        sns.kdeplot(FS1, shade=True, color="b", label='FS1')
        sns.kdeplot(FS2, shade=True, color="g", label='FS2')
        plt.axvline(x=score, color='r', linestyle='--', 
                    label='RENT prediction score')
        plt.legend()
        plt.title('Feasibility Study')

class RENT_Regression(RENT_Base):
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
    
    
    classifier: str, options: 'logreg' for logistic regression
                       'linearSVC' for linear support vector classifier
    
    K: int, number of unique train-test splits
    
    verbose: print something
    
    OUTPUT
    ------
    None 
    """
    
    def __init__(self, data, target, feat_names=[], parameter_selection=True,
                 C=[1,10], l1_ratios = [0.6],
                 poly='OFF', testsize_range=(0.2, 0.6),
                 K=5,  verbose = 0):
        
        # Print parameters for checking
        print('data dimension:', np.shape(data), ' data type:', type(data))
        print('target dimension', np.shape(target))
        print('regularization parameters C:', C)
        print('elastic net l1_ratios:', l1_ratios)
        print('number of models in ensemble:', K)
        print('verbose:', verbose)


        # Define all objects needed later in methods below
        self.target = target
        self.K = K
        self.feat_names = feat_names
        self.testsize_range = testsize_range
        self.verbose = verbose
        
        
        # Check if data is dataframe and add index information
        if isinstance(data, pd.DataFrame):
            self.indices = data.index
        else:
            self.indices = list(range(data.shape[0]))


        # If no feature names are given, then make some
        if len(self.feat_names) == 0:
            print('No feature names found - automatic generate feature names.')
            
            for ind in range(1, np.shape(data)[1] + 1):
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
            sys.exit('Value for paramter "poly" not regcognised.')
            
            
        if parameter_selection == True:
            self.C, self.l1_ratios = self.parameter_selection(C_params=C, 
                                                              l1_params=l1_ratios)
            self.C = [self.C]
            self.l1_ratios = [self.l1_ratios]
        else:
            self.C = C
            self.l1_ratios = l1_ratios
    
            
    def parameter_selection(self, 
                    C_params, 
                    l1_params, 
                    n_splits=5, 
                    testsize_range=(0.25,0.25)):
        """
        

        Parameters
        ----------
        C_params : TYPE
            DESCRIPTION.
        l1_params : TYPE
            DESCRIPTION.
        n_splits : TYPE, optional
            DESCRIPTION. The default is 5.
        testsize_range : TYPE, optional
            DESCRIPTION. The default is (0.25,0.25).

        Returns
        -------
        None.

        """
            
        # if len(np.shape(labels)) == 2:
        #     labels = labels.squeeze()
        
        skf = KFold(n_splits=n_splits, random_state=0, shuffle=True)
        
        scores_df = pd.DataFrame(np.zeros, index=l1_params, columns=C_params)
        zeroes_df = pd.DataFrame(np.zeros, index=l1_params, columns=C_params)
        
        
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
            for reg in C_params:
                scores = list()
                zeroes = list()
                for train, test in skf.split(self.data, self.target):
                    # Find those parameters that are 0
                    sc = StandardScaler()
                    train_data = sc.fit_transform(self.data.iloc[train,:])
                    train_target = self.target[train]
                    test_data_split = sc.transform(self.data.iloc[test,:])
                    test_target = self.target[test]
                    sgd =  ElasticNet(alpha=1/reg, l1_ratio=l1,
                                       max_iter=5000, random_state=0, \
                                       fit_intercept=False).\
                                       fit(train_data, train_target)
                                       
                    mod_coef = sgd.coef_.reshape(1, len(sgd.coef_))
                    params = np.where(mod_coef != 0)[1]
        
                    # if there are parameters != 0, build a predicion model and
                    # find best parameter combination w.r.t. scoring
                    if len(params) == 0:
                        scores.append(np.nan)
                        zeroes.append(np.nan)
                    else:
                        zeroes.append((len(self.data.columns)-len(params))\
                                      /len(self.data.columns))
                        
                        sc = StandardScaler()
                        train_data_1 = sc.fit_transform(train_data[:,params])
                        test_data_1 = sc.transform(test_data_split[:, params])
                        model = LinearRegression().\
                                fit(train_data_1, train_target)
                        scores.append(r2_score(test_target, \
                                        model.predict(test_data_1)))
                                
                scores_df.loc[l1, reg] = np.nanmean(scores)
                zeroes_df.loc[l1, reg] = np.nanmean(zeroes)
    
        
        Parallel(n_jobs=-1, verbose=0, backend="threading")(
             map(delayed(run_parallel), l1_params))  
    
        normed_scores = (scores_df-np.nanmin(scores_df.values))\
        /(np.nanmax(scores_df.values)-np.nanmin(scores_df.values))
        normed_zeroes = (zeroes_df-np.nanmin(zeroes_df.values))\
        /(np.nanmax(zeroes_df.values)-np.nanmin(zeroes_df.values))
        
        combination = (normed_scores ** -1 + normed_zeroes ** -1) ** -1
        best_combination_row, best_combination_col = np.where(combination == \
                                                  np.nanmax(combination.values))
        best_l1 = combination.index[np.nanmax(best_combination_row)]
        best_C = combination.columns[np.nanmin(best_combination_col)]
        
        return(best_C, best_l1)
            
    def run_parallel(self, K):
        """
        Parallel computation of for loops. Parallelizes the number of models (K)
        as this is the parameter with most varying values.
        
        INPUT
        -----
        K: range of train-test splits
        
        OUTPUT
        ------
        None 
        """

        # Loop through all C 
        for C in self.C:
            # Loop through requested number of tt splits
            for l1 in self.l1_ratios:
                
                X_train, X_test, y_train, y_test = train_test_split(
                          self.data, self.target, 
                          test_size=self.random_testsizes[K],
                          random_state=K)
                
#                self.train_sets.append(X_train)
                self.X_test = X_test
                
                # Initialise standard scaler and compute mean and STD from 
                # training data. 
                # Transform train and test dataset
                sc = StandardScaler()
                sc.fit(X_train)
                X_train_std = sc.transform(X_train)
                X_test_std = sc.transform(X_test)
                
                if self.verbose > 0:
                    print('l1 = ', l1, 'C = ', C, ', TT split = ', K)

                model = ElasticNet(alpha=1/C, l1_ratio=l1,
                                       max_iter=5000, random_state=0, \
                                       fit_intercept=False).\
                                       fit(X_train_std, y_train)

                # Get all weights (coefficients). Those that were selected
                # are non-zero, otherwise zero
                mod_coef = model.coef_.reshape(1, len(model.coef_))
                self.weight_dict[(C, l1, K)] = mod_coef
                self.weight_list.append(mod_coef)
                
                pred = model.predict(X_test_std)
                abs_error_df = pd.DataFrame({'abs error': abs(y_test-pred)})
                abs_error_df.index = X_test.index
                self.predictions_abs_errors[(C, l1, K)] = abs_error_df

                score = r2_score(y_test,pred)
                self.score_dict[(C, l1, K)] = score
                self.score_list.append(score)

    
    def train(self):
        """
        This method trains C * l1_ratio * K models in total. The number
        of models using the same hyperparamter is K.
        For each model elastic net regularisation is applied for variable 
        selection.
         
        
        INPUT
        -----
        None
        
        OUTPUT
        ------
        None 
        """
        #check if this is needed and does what it should (probably doesn't work because of parallelization)
        np.random.seed(0)
        self.random_testsizes = np.random.uniform(self.testsize_range[0],
                                                  self.testsize_range[1], 
                                                  self.K)
        
        # Initiate a dictionary holding coefficients for each model. Keys are
        # (C, K, num_w_init)
        self.weight_dict = {}
        
        # Initiate a dictionary holding computed performance metric for each
        self.score_dict = {}


        # Collect all coefs in a list that will be converted to a numpy array
        self.weight_list = []
        
        # Collect all coefs in a list that will be converted to a numpy array
        self.score_list = []
        
        # Initialize a dictionary to predict incorrect labels
        self.predictions_abs_errors = {}

        # stop runtime
        start = time.time()
        # Call parallelization function 
        Parallel(n_jobs=-1, verbose=0, backend='threading')(
             map(delayed(self.run_parallel), range(self.K)))  
        ende = time.time()
        self._runtime = '{:5.3f}s'.format(ende-start)
        
        # find best parameter setting and matrices
        result_list=[]
        for l1 in self.l1_ratios:
            for C in self.C:
                spec_result_list = []
                for k in self.score_dict.keys():
                    if k[0] == C and k[1] ==l1:
                        spec_result_list.append(self.score_dict[k])
                result_list.append(spec_result_list)
    
        means=[]
        for r in range(len(result_list)):
            means.append(np.mean(result_list[r]))

        self._scores_df = pd.DataFrame(np.array(means).reshape(\
                                  len(self.l1_ratios), \
                                  len(self.C)), \
        index= self.l1_ratios, columns = self.C)
        
        self._zeroes_df = pd.DataFrame(index = self.l1_ratios,\
                                   columns=self.C)
        for l1 in self.l1_ratios:
            for C in self.C:
                count = 0
                for K in range(self.K):
                    nz = \
                    len(np.where(pd.DataFrame(self.weight_dict[(C, l1, K)\
])==0)[0])
                    count = count + nz / len(self.feat_names)
                count = count / (self.K)
                self._zeroes_df.loc[l1, C] = count
                
        if len(self.C)>1 or len(self.l1_ratios)>1:
            normed_scores = pd.DataFrame((self._scores_df-np.nanmin(
                self._scores_df.values))/ \
                (np.nanmax(self._scores_df.values)-np.nanmin(
                    self._scores_df.values)))
            normed_zeroes = pd.DataFrame((self._zeroes_df-np.nanmin(
                self._zeroes_df.values))/ \
                (np.nanmax(self._zeroes_df.values)-np.nanmin(
                    self._zeroes_df.values)))
            normed_zeroes = normed_zeroes.astype('float')
    
            self._combination = (normed_scores.copy().applymap(self.inv) + \
                                        normed_zeroes.copy().applymap(
                                            self.inv)).applymap(self.inv)
        else:
            self._combination = (self._scores_df.copy().applymap(self.inv) + \
                                 self._zeroes_df.copy().applymap(
                                     self.inv)).applymap(self.inv)
        
        self._scores_df.columns.name = 'Scores'
        self._zeroes_df.columns.name = 'Zeroes'
        self._combination.columns.name = 'Harmonic Mean'
        
        best_row, best_col  = np.where(
            self._combination == np.nanmax(self._combination.values))
        self._best_l1_ratio = self._combination.index[np.nanmax(best_row)]
        self._best_C = self._combination.columns[np.nanmin(best_col)]
    
    def summary_objects(self):
        """
        

        Returns
        -------
        None.

        """
        self.incorrect_labels = pd.DataFrame({'# test':np.repeat\
                                      (0, np.shape(self.data)[0]),
                                      'average abs error':np.repeat\
                                      (0, np.shape(self.data)[0])})
        self.incorrect_labels.index=self.indices.copy()
        
        specific_predictions = []
        for K in range(self.K):
            specific_predictions.append(self.predictions_abs_errors[
                (self._best_C, self._best_l1_ratio, K)])
        self._histogram_data = pd.concat(specific_predictions, axis=1)
        self._histogram_data.columns = ['K({0})'.format(x+1) \
                                        for x in range(
                                                self._histogram_data.shape[1])]
        count = self._histogram_data.count(axis=1)
        avg_abs_error = np.nanmean(self._histogram_data, axis=1)
        
        summary_df = pd.DataFrame({'count': count, 'avgerror': avg_abs_error})
        summary_df.index = self._histogram_data.index
        
        self.incorrect_labels.loc[summary_df.index,'# test'] = \
        summary_df.loc[:,'count']
        self.incorrect_labels.loc[summary_df.index,'average abs error'] = \
        summary_df.loc[:,'avgerror']
        
        self.incorrect_labels.iloc[ \
            np.where(self.incorrect_labels.iloc[:,0] == 0)[0]] = np.nan

        return self.incorrect_labels
                
    def get_object_errors(self):
        """
        

        Returns
        -------
        None.

        """
        return self._histogram_data
        
    def plot_object_errors(self, object_id, binning='auto', lower=0,
                                  upper=100, kde=False, norm_hist=False):
        """
        Histograms of predicted probabilities. Check scalings etc.

        Parameters
        ----------
        object_id : TYPE
            DESCRIPTION.
        binning : TYPE, optional
            DESCRIPTION. The default is 'auto'.
        lower : TYPE, optional
            DESCRIPTION. The default is 0.
        upper : TYPE, optional
            DESCRIPTION. The default is 1.
        kde : TYPE, optional
            DESCRIPTION. The default is False.
        norm_hist : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        # different binning schemata
        # https://www.answerminer.com/blog/binning-guide-ideal-histogram
        for obj in object_id:
            fig, ax = plt.subplots()
            data = self._histogram_data.loc[obj,:].dropna()
            if binning == "auto":
                bins = None
            if binning == "rice":
                bins = math.ceil(2 * len(data) ** (1./3.))
            if binning == "sturges":
                bins = math.ceil(math.log(len(data), 2)) + 1
        
            sns.set_style("white")
            ax=sns.distplot(data, 
                            bins=bins, 
                            color = 'darkblue',
                            hist_kws={'edgecolor':'darkblue'},
                            kde_kws={'linewidth': 3},
                            kde=kde,
                            norm_hist=norm_hist)
            ax.set(xlim=(lower, upper))
            
            if norm_hist == False:
                ax.set_ylabel('absolute frequencies', fontsize=10)
                ax.set_xlabel('Absolute Error', fontsize=10)
            else:
                ax.set_ylabel('frequencies')
                ax.set_xlabel('Absolute Error')
            ax.set_title('Object: {0}')
    
    def feasibility_study(self, test_data, test_labels, K_feas, alpha=0.05):
        """
        Feasibiliyt study as in paper. p-value has to be added.

        Parameters
        ----------
        test_data : TYPE
            DESCRIPTION.
        test_labels K_feas : TYPE
            DESCRIPTION.
        metric : TYPE, optional
            DESCRIPTION. The default is 'R2'.

        Returns
        -------
        None.

        """
        
        sc = StandardScaler()
        train_RENT = sc.fit_transform(self.data.iloc[:,self.sel_var])
        test_RENT = sc.transform(test_data.loc[:, self.sel_var])

        model = LinearRegression().fit(train_RENT,self.target)
        score = r2_score(test_labels, model.predict(test_RENT))
        # FS1
        FS1 = []
        for K in range(K_feas):
            # Randomly select features (# features = # RENT features selected)
            columns = np.random.RandomState(seed=K).choice(
                range(0,len(self.data.columns)),
                                    len(self.sel_var))
            sc = StandardScaler()
            train_FS1 = sc.fit_transform(self.data.iloc[:, columns])
            test_FS1 = sc.transform(test_data.iloc[:, columns])

            model = LinearRegression().fit(train_FS1,self.target)

            FS1.append(r2_score(test_labels, model.predict(test_FS1)))
            
        p_value_FS1 = sum(FS1 > score) / len(FS1)
        print("p-value average score random feature drawing: ", p_value_FS1)
        if p_value_FS1 <= alpha:
            print('With a significancelevel of ',alpha,' H0 is rejected.')
        else:
            print('With a significancelevel of ',alpha,' H0 is accepted.')
        
        # FS2
        sc = StandardScaler()
        test_data.columns = self.data.columns
        FS2 = []
        
        train_FS2 = sc.fit_transform(self.data.iloc[:,self.sel_var])
        test_FS2 = sc.transform(test_data.loc[:, self.sel_var])

        model = LinearRegression().fit(train_FS2,self.target)
        for K in range(K_feas):
            FS2.append(r2_score(
                    np.random.RandomState(seed=K).permutation(test_labels),\
                    model.predict(test_FS2)))

        p_value_FS2 = sum(FS2 > score) / len(FS2)
        print("p-value score permutation of test labels: ", p_value_FS2)
        if p_value_FS2 <= alpha:
            print('With a significancelevel of ', alpha, ' H0 is rejected.')
        else:
            print('With a significancelevel of ', alpha, ' H0 is accepted.')
            
        sns.kdeplot(FS1, shade=True, color="b", label='FS1')
        sns.kdeplot(FS2, shade=True, color="g", label='FS2')
        plt.axvline(x=score, color='r', linestyle='--', 
                    label='RENT prediction score')
        plt.legend()
        plt.title('Feasibility Study')
    