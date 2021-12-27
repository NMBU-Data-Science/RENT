#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:51:26 2021

@author: anna
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
                            matthews_corrcoef, r2_score, accuracy_score, \
                            log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from scipy.stats import t


class RENT_Base(ABC):
    """
    The constructor initializes common variables of RENT_Classification and RENT_Regresson.
    Initializations that are specific for classification or regression are described in 
    detail in RENT for binary classification and RENT for regression, respectively.
    
    PARAMETERS
    -----
    data: <numpy array> or <pandas dataframe>
        Dataset on which feature selection shall be performed. 
        Variable types must be numeric or integer.
    target: <numpy array> or <pandas dataframe>
        Response variable of data.
    feat_names : <list>
        List holding feature names. Preferably a list of string values. 
        If empty, feature names will be generated automatically. 
        Default: ``feat_names=[]``.
    C : <list of int or float values>
        List with regularisation parameters for ``K`` models. The lower,
        the stronger the regularization is. Default: ``C=[1,10]``.
    l1_ratios : <list of int or float values>
        List holding ratios between l1 and l2 penalty. Values must be in [0,1]. For
        pure l2 use 0, for pure l1 use 1. Default: ``l1_ratios=[0.6]``.
    autoEnetParSel : <boolean>
        Cross-validated elastic net hyperparameter selection.
            - ``autoEnetParSel=True`` : peform a cross-validation pre-hyperparameter\
                search, such that RENT runs only with one hyperparamter setting.
            - ``autoEnetParSel=False`` : perform RENT with each combination of ``C`` \
                and ``l1_ratios``. Default: ``autoEnetParSel=True``.
    BIC : <boolean>
        Use the Bayesian information criterion to select hyperparameters.
            - ``BIC=True`` : use BIC to select RENT hyperparameters.
            - ``BIC=False``: no use of BIC.   
    poly : <str> 
        Create non-linear features. Default: ``poly='OFF'``.
            - ``poly='OFF'`` : no feature interaction.
            - ``poly='ON'`` : feature interaction and squared features (2-polynoms).
            - ``poly='ON_only_interactions'`` : only feature interactions, \
                no squared features.
    testsize_range : <tuple float>
         Inside RENT, ``K`` models are trained, where the testsize defines the 
         proportion of train data used for testing of a single model. The testsize 
         can either be randomly selected inside the range of ``testsize_range`` 
         for each model or fixed by setting the two tuple entries to the same value. 
         The tuple must be in range (0,1). Default: ``testsize_range=(0.2, 0.6)``.
    K : <int>
        Number of unique train-test splits. Default ``K=100``.
    scale : <boolean>
        Columnwise standardization each of the K train datasets. Default ``scale=True``.
    random_state : <None or int>
        Set a random state to reproduce your results. Default: ``random_state=None``.
            - ``random_state=None`` : no random seed. 
            - ``random_state={0,1,2,...}`` : random seed set.       
    verbose : <int>
        Track the train process if value > 1. If ``verbose = 1``, only the overview
        of RENT input will be shown. Default: ``verbose=0``.
    """
    __slots__=["_data", "_target", "_feat_names", "_C", "_l1_ratios", "_autoEnetParSel",
               "_BIC", "_poly", "_testsize_range", "_K", "_scale", "_random_state",
               "_verbose", "_summary_df", "_score_dict", "_BIC_df", "_best_C",
               "_best_l1_ratio", "_indices", "_runtime", "_scores_df", "_combination", 
               "_zeros", "_perc", "_self_var", "_X_test", "_zeros_df","_sel_var",
               "_incorrect_labels", "_pp_data"]

    def __init__(self, data, target, feat_names=[], C=[1,10], l1_ratios = [0.6],
                 autoEnetParSel=True, BIC=False, poly='OFF',testsize_range=(0.2, 0.6), 
                 K=100, scale = True, random_state = None, verbose = 0):

        if any(c < 0 for c in C):
            sys.exit('C values must not be negative!')
        if any(l < 0 for l in l1_ratios) or any(l > 1 for l in l1_ratios):
            sys.exit('l1 ratios must be in [0,1]!')
        if autoEnetParSel not in [True, False]:
            sys.exit('autoEnetParSel must be True or False!')
        if BIC not in [True, False]:
            sys.exit('BIC must be True or False!')
        if scale not in [True, False]:
            sys.exit('scale must be True or False!')
        if poly not in ['ON', 'ON_only_interactions', 'OFF']:
            sys.exit('Invalid poly parameter!')
        if K<=0:
            sys.exit('Invalid K!')
        if K<10:
            # does not show warning...
            warnings.warn('Attention: K is very small!', DeprecationWarning)
        if len(target.shape) == 2 :
            target = target.values

        # Print parameters if verbose = True
        if verbose == 1:
            print('data dimension:', np.shape(data), ' data type:', type(data))
            print('target dimension:', np.shape(target))
            print('regularization parameters C:', C)
            print('elastic net l1_ratios:', l1_ratios)
            print('poly:', poly)
            print('number of models in ensemble:', K)
            print('random state:', random_state)
            print('verbose:', verbose)


        # Define all objects needed later in methods below
        self._target = target
        self._K = K
        self._feat_names = feat_names
        self._testsize_range = testsize_range
        self._scale = scale
        self._verbose = verbose
        self._autoEnetParSel = autoEnetParSel
        self._BIC = BIC
        self._random_state = random_state
        self._poly = poly
        
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, list):
                self._indices = data.index
            else:
                data.index = list(data.index)
                self._indices = data.index
        else:
            self._indices = list(range(data.shape[0]))

        if isinstance(self._target, pd.Series):
            self._target.index = self._indices

        # If no feature names are given, create some
        if len(self._feat_names) == 0:
            print('No feature names found - automatic generate feature names.')

            for ind in range(1, np.shape(data)[1] + 1):
                self._feat_names.append('f' + str(ind))

        # Extend data if poly was set to 'ON' or 'ON_only_interactions'
        if self._poly == 'ON':
            self._polynom = PolynomialFeatures(interaction_only=False, \
                                         include_bias=False)
            self._data = self._polynom.fit_transform(data)
            polynom_comb = list(combinations_with_replacement(self._feat_names,\
                                                              2))
            polynom_feat_names = []
            # Construct a new name for squares and interactions
            for item in polynom_comb:
                if item[0] == item[1]:
                    name = item[0] + '^2'
                else:
                    name = item[0] + '*' + item[1]
                polynom_feat_names.append(name)

            flist = list(self._feat_names)
            flist.extend(polynom_feat_names)
            self._feat_names = flist
            self._data = pd.DataFrame(self._data)
            self._data.index = self._indices
            self._data.columns = self._feat_names

        elif self._poly == 'ON_only_interactions':
            self._polynom = PolynomialFeatures(interaction_only=True,\
                                         include_bias=False)
            self._data = self._polynom.fit_transform(data)

            polynom_comb = list(combinations(self._feat_names, 2))
            polynom_feat_names = []

            # Construct a new name for squares and interactions
            for item in polynom_comb:
                name = item[0] + '*' + item[1]
                polynom_feat_names.append(name)

            flist = list(self._feat_names)
            flist.extend(polynom_feat_names)
            self._feat_names = flist
            self._data = pd.DataFrame(self._data)
            self._data.index = self._indices
            self._data.columns = self._feat_names

        elif self._poly == 'OFF':
            self._data = pd.DataFrame(data)
            self._data.index=self._indices
            self._data.columns = self._feat_names

        else:
            sys.exit('Value for paramter "poly" not regcognised.')
        

        if self._autoEnetParSel == True:
            if self._BIC == False:
                self._C, self._l1_ratios = self._par_selection(C=C, 
                                                            l1_ratios=l1_ratios)
            else:
                self._C, self._l1_ratios = self._par_selection_BIC(C=C, 
                                                            l1_ratios=l1_ratios)
            self._C = [self._C]
            self._l1_ratios = [self._l1_ratios]
        else:
            self._C = C
            self._l1_ratios = l1_ratios
    
    @abstractmethod
    def run_parallel(self, K):
        pass

    @abstractmethod
    def _par_selection(self, C_params, l1_params, n_splits, testsize_range):
        pass
    
    @abstractmethod
    def _par_selection_BIC(self, C_params, l1_params, n_splits, testsize_range):
        pass

    @abstractmethod
    def get_summary_objects(self):
        pass
    
    @abstractmethod
    def _prepare_validation_study(self, test_data, test_labels, num_drawings, 
                                  num_permutations, metric='mcc', alpha=0.05):
        pass

    def train(self):
        """
        If ``autoEnetParSel=False``, this method trains ``K`` * ``len(C)`` 
        * ``len(l1_ratios)`` models in total. 
        The number of models using the same hyperparamters is ``K``.
        Otherwise, if the best parameter combination is selected with 
        cross-validation, only ``K`` models are trained.
        For each model elastic net regularisation is applied for feature selection. 
        Internally, ``train()`` calls the ``run_parallel()`` function for classification 
        or regression, respectively.
        """
        np.random.seed(0)
        self._random_testsizes = np.random.uniform(self._testsize_range[0],
                                                  self._testsize_range[1],
                                                  self._K)

        # Initiate dictionaries. Keys are (C, K, num_w_init)
        self._weight_dict = {}
        self._score_dict = {}
        self._weight_list = []
        self._score_list = []

        # stop runtime
        start = time.time()
        # Call parallelization function
        Parallel(n_jobs=-1, verbose=0, backend='threading')(
             map(delayed(self.run_parallel), range(self._K)))
        ende = time.time()
        self._runtime = ende-start

        # find best parameter setting and matrices
        result_list=[]
        for l1 in self._l1_ratios:
            for C in self._C:
                spec_result_list = []
                for k in self._score_dict.keys():
                    if k[0] == C and k[1] ==l1:
                        spec_result_list.append(self._score_dict[k])
                result_list.append(spec_result_list)

        means=[]
        for r in range(len(result_list)):
            means.append(np.mean(result_list[r]))

        self._scores_df = pd.DataFrame(np.array(means).reshape(\
                                  len(self._l1_ratios), \
                                  len(self._C)), \
        index= self._l1_ratios, columns = self._C)

        self._zeros_df = pd.DataFrame(index = self._l1_ratios,\
                                   columns=self._C)
        for l1 in self._l1_ratios:
            for C in self._C:
                count = 0
                for K in range(self._K):
                    nz = \
                    len(np.where(pd.DataFrame(self._weight_dict[(C, l1, K)\
])==0)[0])
                    count = count + nz / len(self._feat_names)
                count = count / (self._K)
                self._zeros_df.loc[l1, C] = count

        if len(self._C)>1 or len(self._l1_ratios)>1:
            normed_scores = pd.DataFrame(self._min_max(
                self._scores_df.copy().values))
            normed_zeros = pd.DataFrame(self._min_max(
                self._zeros_df.copy().values))
            normed_zeros = normed_zeros.astype('float')
            self._combination = 2 * ((normed_scores.copy().applymap(self._inv) + \
                                        normed_zeros.copy().applymap(
                                            self._inv)).applymap(self._inv))
        else:
            self._combination = 2 * ((self._scores_df.copy().applymap(self._inv) + \
                                 self._zeros_df.copy().applymap(
                                     self._inv)).applymap(self._inv))
        self._combination.index = self._scores_df.index.copy()
        self._combination.columns = self._scores_df.columns.copy()

        self._scores_df.columns.name = 'Scores'
        self._zeros_df.columns.name = 'Zeros'
        self._combination.columns.name = 'Harmonic Mean'

        best_row, best_col  = np.where(
            self._combination == np.nanmax(self._combination.values))
        self._best_l1_ratio = self._combination.index[np.nanmax(best_row)]
        self._best_C = self._combination.columns[np.nanmin(best_col)]

    def select_features(self, tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975):
        """
        Selects features based on the cutoff values for tau_1_cutoff, 
        tau_2_cutoff and tau_3_cutoff.
        
        Parameters
        ----------
        tau_1_cutoff : <float>
            Cutoff value for tau_1 criterion. Choose value between 0 and
            1. Default: ``tau_1=0.9``.
        tau_2_cutoff : <float>
            Cutoff value for tau_2 criterion. Choose value between 0 and
            1. Default:``tau_2=0.9``.
        tau_3_cutoff : <float>
            Cutoff value for tau_3 criterion. Choose value between 0 and
            1. Default: ``tau_3=0.975``.
            
        Returns
        -------
        <numpy array>
            Array with selected features.
        """
        if not hasattr(self, '_best_C'):
            sys.exit('Run train() first!')

        weight_list = []
        #Loop through all K models
        for K in range(self._K):
            weight_list.append(self._weight_dict[(self._best_C,
                                                 self._best_l1_ratio,
                                                 K)])
        weight_array = np.vstack(weight_list)

        #Compute results based on weights
        counts = np.count_nonzero(weight_array, axis=0)
        self._perc = counts / len(weight_list)
        means = np.mean(weight_array, axis=0)
        stds = np.std(weight_array, axis=0)
        signum = np.apply_along_axis(self._sign_vote, 0, weight_array)
        t_test = t.cdf(
            abs(means / np.sqrt((stds ** 2) / len(weight_list))), \
                (len(weight_list)-1))

        # Conduct a dataframe that stores the results for the criteria
        summary = np.vstack([self._perc, signum, t_test])
        self._summary_df = pd.DataFrame(summary)
        self._summary_df.index = ['tau_1', 'tau_2', 'tau_3']
        self._summary_df.columns = self._feat_names

        self._sel_var = np.where(
                (self._summary_df.iloc[0, :] >= tau_1_cutoff) &
                (self._summary_df.iloc[1, :] >= tau_2_cutoff) &
                (self._summary_df.iloc[2, :] >= tau_3_cutoff\
                            ))[0]
        
        #if len(self._sel_var) == 0:
        #    warnings.warn("Attention! Thresholds are too restrictive - no features selected!")
        return self._sel_var
    
    def BIC_cutoff_search(self, parameters):
        """
        Compute the Bayesian information criterion for each combination of tau1, tau2 and tau3.
        
        PARAMETERS
        -----
        parameters: <dict> or
            Cutoff parameters to evaluate.
        Returns
        -------
        <numpy array>
            Array wth the BIC values.
        """
        sc = StandardScaler()
        # Bayesian information criterion
        BIC = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                                len(parameters['t3'])))
        # grid search t1, t2, t3
        for i, t1 in enumerate(parameters['t1']):
            for j, t2 in enumerate(parameters['t2']):
                for k, t3 in enumerate(parameters['t3']):
                    sel_feat = self.select_features(t1, t2, t3)

                    train_data = sc.fit_transform(self._data.iloc[:,sel_feat])
                    lr = LogisticRegression().fit(train_data, self._target)
                    num_params = len(np.where(lr.coef_ != 0)[1]) + 1
                    pred_proba = lr.predict_proba(train_data)
                    pred = lr.predict(train_data)
                    
                    log_lik = log_loss(y_true=self._target, y_pred=pred_proba, normalize=False)
                    B = 2 * log_lik + np.log(len(pred)) * num_params
                    BIC[i,j,k] = B
                    
        return BIC

    def get_summary_criteria(self):
        """
        Summary statistic of the selection criteria tau_1, tau_2 and 
        tau_3 (described in ``select_features()``)
        for each feature. All three criteria are in [0,1] .
        
        RETURNS
        -------
        <pandas dataframe>
            Matrix where rows represent selection criteria and 
            columns represent features.
        """
        if not hasattr(self, '_summary_df'):
            sys.exit('Run select_features() first!')
        return self._summary_df

    def get_weight_distributions(self, binary = False):
        """
        In each of the ``K`` models, feature weights are fitted, i.e. 
        an individiual weight is assigned feature 1 for model 1, 
        model 2, up to model ``K``. This method returns the weight 
        for every feature and model (1:``K``) combination.
        
        PARAMETERS
        ----------
        binary : <boolean>
            Default: ``binary=False``.
                - ``binary=True`` : binary matrix where entry is 1 \
                    for each weight unequal to 0.
                - ``binary=False`` : original weight matrix.
                
        RETURNS
        -------
        <pandas dataframe>
            Weight matrix. Rows represent models (1:K), 
            columns represents features.
        """
        if not hasattr(self, '_weight_dict'):
            sys.exit('Run train() first!')

        weights_df = pd.DataFrame()
        for k in self._weight_dict.keys():
            if k[0] == self._best_C and k[1] == self._best_l1_ratio:
                weights_df = weights_df.append( \
                        pd.DataFrame(self._weight_dict[k]))
        weights_df.index = ['mod {0}'.format(x+1) for x in range(self._K)]
        weights_df.columns = self._feat_names
        
        if binary == True:
            return((weights_df != 0).astype(np.int_))
        else:
            return(weights_df)

    def get_scores_list(self):
        """
        Prediction scores over the ``K`` models.
        RETURNS
        -------
        <list>
            Scores list.
        """
        scores_list = []
        for k in self._score_dict.keys():
            if k[0] == self._best_C and k[1] == self._best_l1_ratio:
                scores_list.append(self._score_dict[k])
        return scores_list

    def get_enetParam_matrices(self):
        """
        Three pandas data frames showing result for all combinations
        of ``l1_ratio`` and ``C``.
        
        RETURNS
        -------
        <list> of <pandas dataframes>
            - dataFrame_1: holds average scores for \
                predictive performance.
            - dataFrame_2: holds average percentage of \
                how many feature weights were set to zero.
            - dataFrame_3: holds harmonic means between \
                dataFrame_1 and dataFrame_2.
        """
        if not hasattr(self, '_weight_dict'):
            sys.exit('Run train() first!')
        return self._scores_df, self._zeros_df, self._combination

    def get_cv_matrices(self):
        """
        Three pandas data frames showing cross-validated result for all combinations
        of ``C`` and ``l1_ratio`` . Only applicable if ``autoEnetParSel=True``.
        
        RETURNS
        -------
        <list> of <pandas dataframes>
            - dataFrame_1: average scores for predictive performance. \
                The higher the score, the better the parameter combination. 
            - dataFrame_2: average percentage of how many feature weights \
                are set to zero. The higher the average percentage, the stronger \
                    the feature selection with the corresponding paramter combination.
            - dataFrame_3: harmonic means between normalized dataFrame_1 and normalized \
                dataFrame_2. The parameter combination with the highest \
                    harmonic mean is selected.
        """
        if self._autoEnetParSel == True and self._BIC ==False:
            return self._scores_df_cv, self._zeros_df_cv, self._combination_cv
        else:
            print("autoEnetParSel=False or BIC=True - parameters have not been selected with cross-validation.")

    def get_BIC_matrix(self):
        """
        Dataframe with BIC value for each combination of ``C`` and ``11_ratio``.
        RETURNS
        -------
        <pandas dataframes>
            Dataframe of BIC values.
        """
        if self._autoEnetParSel == True and self._BIC ==True:
            return self._BIC_df
        else:
            print("BIC=False - parameters have not been selected with BIC.")
    
    def get_enet_params(self):
        """
        Get current hyperparameter combination of ``C`` and ``l1_ratio`` that 
        is used in RENT analyses. By default it is the best combination found. 
        If `autoEnetParSel=False` the user can change the combination 
        with ``set_enet_params()``. 
        
        RETURNS
        -------
        <tuple>
            A tuple (C, l1_ratio).
        """
        if not hasattr(self, '_best_C'):
            sys.exit('Run train() first!')
        return self._best_C, self._best_l1_ratio
    
    def get_runtime(self):
        """
        Total RENT training time in seconds.
        
        RETURNS
        -------
        <numeric value>
            Time.
        """
        return self._runtime

    def set_enet_params(self, C, l1_ratio):
        """
        Set hyperparameter combination of ``C`` and ``l1_ratio``, 
        that is used for analyses. Only useful if ``autoEnetParSel=False``.
        
        PARAMETERS
        ----------
        C: <float>
            Regularization parameter.
        l1_ratio: <float>
            l1 ratio with value in [0,1]. 
        """
        
        if (C not in self._C) | (l1_ratio not in self._l1_ratios):
            sys.exit('No weights calculated for this combination!')
        self._best_C = C
        self._best_l1_ratio = l1_ratio

    def plot_selection_frequency(self):
        """
        Barplot of tau_1 value for each feature.
        """
        if not hasattr(self, '_perc'):
            sys.exit('Run select_features() first!')

        plt.figure(figsize=(10, 7))
        (markers, stemlines, baseline) = plt.stem(self._perc,\
        use_line_collection=True)
        plt.setp(markers, marker='o', markersize=5, color='black',
            markeredgecolor='darkorange', markeredgewidth=0)
        plt.setp(stemlines, color='darkorange', linewidth=0.5)
        plt.show()

    def plot_elementary_models(self):
        """
        Two lineplots where the first curve shows the prediction score over 
        ``K`` models. The second curve plots the percentage of weights set 
        to 0, respectively.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        num_zeros = np.sum(1 - self.get_weight_distributions(binary=True), \
                            axis=1) / len(self._feat_names)
        
        scores = self.get_scores_list()
        data = pd.DataFrame({"num_zeros" : num_zeros, "scores" : scores})
        
        plt.plot(data.num_zeros.values, linestyle='--', marker='o', \
                 label="% zero weights")
        plt.plot(data.scores.values, linestyle='--', marker='o', label="score")
        plt.legend()
        ax.set_xlabel('elementary models')
        plt.title("Analysis of ensemble models")

    
    def plot_object_PCA(self, cl=0, comp1=1, comp2=2, 
                        problem='class', hoggorm=True, 
                        hoggorm_plots=[1,2,3,4,6], sel_vars=True):
        """
        PCA analysis. For classification problems, PCA can be computed either 
        on a single class separately or on both classes. Different coloring 
        possibilities for the scores are provided.
        Besides scores, loadings, correlation loadings, biplot, and explained 
        variance plots are available. 
        
        Parameters
        ----------
        cl : <int>, <str>
            Perform PCA on cl. Default: ``cl=0``.
                - ``cl=0``: Class 0.
                - ``cl=1``: Class 1.
                - ``cl='both'``: All objects (incorrect predictions coloring).
                - ``cl='continuous'``: All objects (gradient coloring). \
                    For classification problems, this is the only valid option.
        comp1 : <int>
            First PCA component to plot. Default: ``comp1=1``.            
        comp2 : <int>
            Second PCA component to plot. Default: ``comp2=2``.  
        problem : <str>
            Classification or regression problem. Default: ``problem='class'``.
                - ``problem='class'``: Classification problem. Can be used with \
                    all possible ``cl`` inputs.
                - ``problem='regression'``: Regression problem. \
                    Can only be used with ``cl='continuous'``.
        hoggorm : <boolean>
            To not use plots from hoggormplot package, set ``hoggorm=False``. \
                Default: ``hoggorm=True``.
        hoggorm_plots : <list>
            Choose which plots from hoggormplot are plotted. Only plots that are \
                relevant for RENT are possible options. ``hoggorm=True`` must be set. \
                    Default: ``hoggorm_plots=[1,2,3,4,6]``.
                - 1: scores plot
                - 2: loadings plot
                - 3: correlation loadings plot
                - 4: biplot
                - 6: explained variance plot
        sel_vars : <boolean>
            Only use the features selected with RENT for PCA. Default: ``sel_vars=True``.            
        """
        if cl not in [0, 1, 'both', 'continuous']:
            sys.exit(" 'cl' must be either 0, 1, 'both' or 'continuous'")
        if problem not in ['class', 'regression']:
            sys.exit(" 'problem' must be either 'class' or 'regression' ")
        if not hasattr(self, '_sel_var'):
            sys.exit('Run select_features() first!')
        if not hasattr(self, '_incorrect_labels'):
            sys.exit('Run get_summary_objects() first!')
        if problem == "regression" and cl != "continuous":
            sys.exit("The input is invalid. For 'problem = regression', 'cl' \
                     must be 'continuous' ")
        # catch if classification with continuous. (check if RENT class or RENT reg)

        if cl != 'continuous':
            dat = pd.merge(self._data, self._incorrect_labels.iloc[:,[1,-1]], \
                                 left_index=True, right_index=True)
            if sel_vars == True:
                variables = list(self._sel_var)
                variables.extend([-2,-1])
        else:
            
            if problem == "regression":
                dat = pd.merge(self._data, self._incorrect_labels.iloc[:,-1], \
                                         left_index=True, right_index=True)
            else:
                obj_mean = pd.DataFrame(np.nanmean( \
                        self.get_object_probabilities(), 1), \
                    index=self._data.index)
                obj_mean.columns = ["pred_means"]
                dat = pd.merge(self._data, obj_mean, \
                                         left_index=True, right_index=True)
            if sel_vars == True:
                variables = list(self._sel_var)
                variables.extend([-1])

        if cl == 'both' or cl == 'continuous':
            if sel_vars == True:
                data = dat.iloc[:,variables]
            else:
                data = dat
        else:
            if sel_vars == True:
                data = dat.iloc[np.where(dat.iloc[:,-2]==cl)[0],variables]
            else:
                data = dat
                
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

        XexplVar = pca_model.X_calExplVar()
        var_comp1 = round(XexplVar[comp1-1], 1)
        var_comp2 = round(XexplVar[comp2-1], 1)


        fig, ax = plt.subplots()
        ax.set_xlabel('comp ' + str(comp1) +' ('+str(var_comp1)+'%)', fontsize=10)
        ax.set_ylabel('comp ' + str(comp2) +' ('+str(var_comp2)+'%)', fontsize=10)
        ax.set_title('Scores plot', fontsize=10)
        ax.set_facecolor('silver')

        # Find maximum and minimum scores along the two components
        xMax = max(scores.iloc[:, (comp1-1)])
        xMin = min(scores.iloc[:, (comp1-1)])

        yMax = max(scores.iloc[:, (comp2-1)])
        yMin = min(scores.iloc[:, (comp2-1)])

        # Set limits for lines representing the axes.
        # x-axis
        if abs(xMax) >= abs(xMin):
            extraX = xMax * .4
            limX = xMax * .3

        elif abs(xMax) < abs(xMin):
            extraX = abs(xMin) * .4
            limX = abs(xMin) * .3

        if abs(yMax) >= abs(yMin):
            extraY = yMax * .4
            limY = yMax * .3

        elif abs(yMax) < abs(yMin):
            extraY = abs(yMin) * .4
            limY = abs(yMin) * .3

        xMaxLine = xMax + extraX; xMinLine = xMin - extraX
        yMaxLine = yMax + extraY; yMinLine = yMin - extraY

        ax.plot([0, 0], [yMaxLine, yMinLine], color='0.4', linestyle='dashed',
                linewidth=3)
        ax.plot([xMinLine, xMaxLine], [0, 0], color='0.4', linestyle='dashed',
                linewidth=3)

        # Set limits for plot regions.
        xMaxLim = xMax + limX; xMinLim = xMin - limX
        yMaxLim = yMax + limY; yMinLim = yMin - limY
        ax.set_xlim(xMinLim, xMaxLim); ax.set_ylim(yMinLim, yMaxLim)

        # plot
        if cl == 0:
            plt.scatter(scores['PC'+str(comp1)], scores['PC'+str(comp2)],
                        c= scores['coloring'], cmap='Greens', marker ="^")
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 0', fontsize=10)
        elif cl == 1:
            plt.scatter(scores['PC'+str(comp1)], scores['PC'+str(comp2)],
                        c= scores['coloring'], cmap='Reds')
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 1', fontsize=10)
        elif cl == 'both':
            zeros = np.where(data.iloc[:,-2]==0)[0]
            ones = np.where(data.iloc[:,-2]==1)[0]

            plt.scatter(scores.iloc[zeros,(comp1-1)],
                        scores.iloc[zeros,(comp2-1)],
                        c= scores.iloc[zeros,-1], cmap='Greens', marker="^",
                        alpha=0.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('% incorrect predicted class 0', fontsize=10)
            plt.scatter(scores.iloc[ones,(comp1-1)],
                        scores.iloc[ones,(comp2-1)],
                        c= scores.iloc[ones,-1], cmap='Reds', alpha=0.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('% incorrect predicted class 1', fontsize=10)

            mlist = []
            col_list = []

            for i in range(len(data.index)):
                if data.iloc[i,-2]==0:
                    mlist.append("^")
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
                elif data.iloc[i,-2]==0 and data.iloc[i,-1]>=50 and data.iloc[i,-1]<100:
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
                plt.scatter(scores.iloc[i,(comp1-1)], scores.iloc[i,(comp2-1)],
                            marker=mlist[i], c=col_list[i])

        elif cl == 'continuous':
            plt.scatter(scores.iloc[:,(comp1-1)], scores.iloc[:,(comp2-1)],
                        c=scores.iloc[:,-1],
                        cmap='YlOrRd')
            cbar = plt.colorbar()
            if problem == "class":
                cbar.set_label('average object prediction', fontsize=10)
            else:
                cbar.set_label('mean absolute error', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        objnames = list(data.index.astype('str'))
        if hoggorm == True:
            if cl != 'continuous':
                hopl.plot(pca_model, plots=hoggorm_plots, comp = [comp1,comp2],
                        objNames=objnames, XvarNames=list(data.columns[:-2]))
            else:
                hopl.plot(pca_model, plots=hoggorm_plots, comp = [comp1,comp2],
                        objNames=objnames, XvarNames=list(data.columns[:-1]))

    def plot_validation_study(self, test_data, test_labels, num_drawings, 
                              num_permutations, metric='mcc', alpha=0.05):
        """
        Two validation studies based on a Student's `t`-test. \
            The null-hypotheses claim that
            - RENT is not better than random feature selection.
            - RENT performs equally well on the real and a randomly permutated target.
            
        If ``poly='ON'`` or ``poly='ON_only_interactions'`` in the RENT initialization, 
        the test data is automatically polynomially transformed.
        
        PARAMETERS
        ----------
        
        test_data : <numpy array> or <pandas dataframe>
            Dataset, used to evalute predictive models in the validation study.
            Must be independent of the data, RENT is computed on.
        test_lables: <numpy array> or <pandas dataframe>
            Response variable of test_data.
        num_drawings: <int>
            Number of independent feature subset drawings for VS1.
        num_permutations: <int>
            Number of independent test_labels permutations for VS2.
        metric: <str>
            The metric to evaluate ``K`` models. Default: ``metric='mcc'``. 
            Only relevant for classification tasks. For regression R2-score is used.
            
                - ``scoring='accuracy'`` :  Accuracy
                - ``scoring='f1'`` : F1-score
                - ``scoring='precision'`` : Precision
                - ``scoring='recall'``: Recall
                - ``scoring='mcc'`` : Matthews Correlation Coefficient
        alpha: <float>
            Significance level for the `t`-test. Default ``alpha=0.05``.
        """
        if not hasattr(self, '_sel_var'):
            sys.exit('Run select_features() first!')

        if self._poly != 'OFF':
            test_data = pd.DataFrame(self._polynom.fit_transform(test_data))
            test_data.columns = self._data.columns
            self._test_data = test_data
        
        score, VS1, VS2 = self._prepare_validation_study(test_data, 
                                                         test_labels, 
                                                         num_drawings, 
                                                         num_permutations,
                            metric='mcc', alpha=0.05)

        heuristic_p_value_VS1 = sum(VS1 > score) / len(VS1)
        T = (np.mean(VS1) - score) / (np.std(VS1,ddof=1) / np.sqrt(len(VS1)))
        print("mean VS1", np.mean(VS1))
        p_value_VS1 = t.cdf(T, len(VS1)-1)
        print("VS1: p-value for average score from random feature drawing: ", 
              p_value_VS1)
        print("VS1: heuristic p-value (how many scores are higher than" +
              " the RENT score): ", heuristic_p_value_VS1)

        if p_value_VS1 <= alpha:
            print('With a significancelevel of ', alpha, ' H0 is rejected.')
        else:
            print('With a significancelevel of ', alpha, ' H0 is accepted.')
        print(' ')
        print('-----------------------------------------------------------')
        print(' ')
   
        heuristic_p_value_VS2 = sum(VS2 > score) / len(VS2)
        print("Mean VS2", np.mean(VS2))
        T = (np.mean(VS2) - score) / (np.std(VS2,ddof=1) / np.sqrt(len(VS2)))
        p_value_VS2 = t.cdf(T, len(VS2)-1)
        print("VS2: p-value for average score from permutation of test labels: ", 
              p_value_VS2)
        print("VS2: heuristic p-value (how many scores are higher"+
              " than the RENT score): ", heuristic_p_value_VS2)
        if p_value_VS2 <= alpha:
            print('With a significancelevel of ', alpha, ' H0 is rejected.')
        else:
            print('With a significancelevel of ', alpha, ' H0 is accepted.')

        plt.figure(figsize=(15, 7))
        sns.kdeplot(VS1, shade=True, color="b", label='VS1')
        sns.kdeplot(VS2, shade=True, color="g", label='VS2')
        plt.axvline(x=score, color='r', linestyle='--',
                    label='RENT prediction score')
        plt.legend(prop={'size': 12})
        plt.ylabel('density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Validation study', fontsize=18)

    def _inv(self, num):
        """
        Invert a numeric value unequal to 0.
        
        PARAMETERS
        ----------
        <float>
            ``num``: numeric value
            
        RETURNS
        -------
        <numeric value>
            Inverted value. 
        """
        if num == 0:
            return np.inf
        elif num == np.inf:
            return 0
        else:
            return num ** -1

    def _sign_vote(self, arr):
        """
        Calculate tau_2.
        
        PARAMETERS
        ----------
        <numpy array>
            ``arr``: Array of numeric values.
            
        RETURNS
        -------
        <numeric value>
            Inverted value. 
        """
        return np.abs(np.sum(np.sign(arr))) / len(arr)

    def _min_max(self, arr):
        """
        Min-max standardization. 
        
        PARAMETERS
        ----------
        <numpy array>
            ``arr``: Array of numeric values.
            
        RETURNS
        -------
        <numpy array>
            1-d array or matrix of higher dimension.
        """
        return (arr-np.nanmin(arr)) / (np.nanmax(arr)-np.nanmin(arr))


class RENT_Classification(RENT_Base):
    """
    This class carries out RENT on a given binary classification dataset. 
    
    PARAMETERS
    ----------
    data : <numpy array> or <pandas dataframe>
        Dataset on which feature selection is performed. \
            Variable types must be numeric or integer.            
    target : <numpy array> or <pandas dataframe>
        Response variable of data.        
    feat_names : <list>
        List holding feature names. Preferably a list of string values. \
            If empty, feature names will be generated automatically. \
                Default: ``feat_names=[]``.                
    C : <list of int or float values>
        List with regularisation parameters for ``K`` models. The lower,
        the stronger the regularization is. Default: ``C=[1,10]``.        
    l1_ratios : <list of int or float values>
        List holding ratios between l1 and l2 penalty. Values must be in [0,1]. \
            For pure l2 use 0, for pure l1 use 1. Default: ``l1_ratios=[0.6]``.            
    autoEnetParSel : <boolean>
        Cross-validated elastic net hyperparameter selection.
            - ``autoEnetParSel=True`` : peform a cross-validation pre-hyperparameter \
                search, such that RENT runs only with one hyperparamter setting.
            - ``autoEnetParSel=False`` : perform RENT with each combination of ``C`` \
                and ``l1_ratios``. Default: ``autoEnetParSel=True``.        
    poly : <str> 
        Create non-linear features. Default: ``poly='OFF'``.
            - ``poly='OFF'`` : no feature interaction.
            - ``poly='ON'`` : feature interaction and squared features (2-polynoms).
            - ``poly='ON_only_interactions'`` : only feature interactions, \
                no squared features.               
    testsize_range : <tuple float>
            Inside RENT, ``K`` models are trained, where the testsize defines the \
                proportion of train data used for testing of a single model. The testsize 
            can either be randomly selected inside the range of ``testsize_range`` for \
                each model or fixed by setting the two tuple entries to the same value. 
            The tuple must be in range (0,1).
            Default: ``testsize_range=(0.2, 0.6)``.            
    scoring : <str>
        The metric to evaluate K models. Default: ``scoring='mcc'``.
            - ``scoring='accuracy'`` :  Accuracy
            - ``scoring='f1'`` : F1-score
            - ``scoring='precision'`` : Precision
            - ``scoring='recall'``: Recall
            - ``scoring='mcc'`` : Matthews Correlation Coefficient            
    classifier : <str>
        Classifier with witch models are trained.
            - ``classifier='logreg'`` : Logistic Regression            
    K : <int>
        Number of unique train-test splits. Default: ``K=100``.        
    scale : <boolean>
        Columnwise standardization of the ``K`` train datasets. \
            Default: ``scale=True``.
    random_state : <None or int>
        Set a random state to reproduce your results. \
            Default: ``random_state=None``.
            - ``random_state=None`` : no random seed. 
            - ``random_state={0,1,2,...}`` : random seed set.        
    verbose : <int>
        Track the train process if value > 1. If ``verbose = 1``, only the overview
        of RENT input will be shown. Default: ``verbose=0``.
        
    RETURNS
    ------
    <class>
        A class that contains the RENT classification model.
    """
    __slots__=["_data", "_target", "_feat_names", "_C", "_l1_ratios", "_autoEnetParSel",
               "_BIC", "_poly", "_testsize_range", "_K", "_scale", "_random_state",
               "_verbose", "_summary_df", "_score_dict", "_BIC_df", "_best_C",
               "_best_l1_ratio", "_indices", "_runtime", "_scores_df", "_combination", 
               "_zeros", "_perc", "_self_var", "_scores_df_cv", "_zeros_df_cv",
               "_combination_cv", "_scoring","_classifier", "_predictions_dict","_probas",
               "_pred_proba_dict", "_random_testsizes", "_weight_dict", "_weight_list", "_score_list"]

    def __init__(self, data, target, feat_names=[], C=[1,10], l1_ratios = [0.6],
                 autoEnetParSel=True, BIC=False, poly='OFF',
                 testsize_range=(0.2, 0.6), scoring='accuracy',
                 classifier='logreg', K=100, scale = True, random_state = None, 
                 verbose = 0):

        super().__init__(data, target, feat_names, C, l1_ratios, 
                         autoEnetParSel, BIC, poly, testsize_range, K, scale, 
                         random_state, verbose)
        
        if scoring not in ['accuracy', 'f1', 'mcc']:
            sys.exit('Invalid scoring!')
        if classifier not in ['logreg', 'linSVC']:
            sys.exit('Invalid classifier')

        if verbose == 1:
            print('classifier:', classifier)
            print('scoring:', scoring)


        # Define all objects needed later in methods below
        self._scoring = scoring
        self._classifier = classifier

    def _par_selection(self,
                        C,
                        l1_ratios,
                        n_splits=5,
                        testsize_range=(0.25,0.25)):
        """
        Preselect best `C` and `l1 ratio` with cross-validation.
        
        PARAMETERS
        ----------
        C: <list of int or float values>
            List holding regularisation parameters for `K` models. The lower, the
            stronger the regularization is.            
        l1_ratios: <list of int or float values>
            List holding ratios between l1 and l2 penalty. Must be in [0,1]. For
            pure l2 use 0, for pure l1 use 1.
        n_splits : <int>
            Number of cross-validation folds. Default: ``n_splits=5``.            
        testsize_range: <tuple float>
            Range of random proportion of dataset to include in test set,
            low and high are floats between 0 and 1. \
                Default: ``testsize_range=(0.2, 0.6)``.
            Testsize can be fixed by setting low and high to the same value.
            
        RETURNS
        -------
        <tuple>
            - First entry: suggested `C` parameter.
            - Second entry: suggested `l1 ratio`.            
        """
        
        skf = StratifiedKFold(n_splits=n_splits, random_state=self._random_state,\
                              shuffle=True)
        scores_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C)
        zeros_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C)
        
        def run_parallel(l1):
            """
            Parallel computation of for ``K`` * ``C`` * ``l1_ratios`` models.
            
            PARAMETERS
            -----
            l1: current l1 ratio in the parallelization framework.
            """
            for reg in C:
                scores = list()
                zeros = list()
                for train, test in skf.split(self._data, self._target):
                    if self._scale == True:
                        sc = StandardScaler()
                        train_data = sc.fit_transform(self._data.iloc[train, :])
                        train_target = self._target[train]
                        test_data_split = sc.transform(self._data.iloc[test, :])
                        test_target = self._target[test]
                    elif self._scale == False:
                        train_data = self._data.iloc[train, :].values
                        train_target = self._target[train]
                        test_data_split = self._data.iloc[test, :].values
                        test_target = self._target[test]

                    sgd = LogisticRegression(penalty="elasticnet", C=reg,
                                             solver="saga", l1_ratio=l1,
                                             random_state=self._random_state)

                    sgd.fit(train_data, train_target)

                    params = np.where(sgd.coef_ != 0)[1]
                    if len(params) == 0:
                        scores.append(np.nan)
                        zeros.append(np.nan)
                    else:
                        zeros.append((len(self._data.columns)-len(params))\
                                      /len(self._data.columns))

                        train_data_1 = train_data[:,params]
                        test_data_1 = test_data_split[:, params]

                        model = LogisticRegression(penalty='none',
                                                   max_iter=8000,
                                                   solver="saga",
                                                   random_state=self._random_state).\
                                fit(train_data_1, train_target)
                        scores.append(matthews_corrcoef(test_target, \
                                        model.predict(test_data_1)))

                scores_df.loc[l1, reg] = np.nanmean(scores)
                zeros_df.loc[l1, reg] = np.nanmean(zeros)

        self._scores_df_cv = scores_df
        self._zeros_df_cv = zeros_df
        self._scores_df_cv.columns.name = 'Scores'
        self._zeros_df_cv.columns.name = 'Zeros'

        Parallel(n_jobs=-1, verbose=1, backend="threading")(
             map(delayed(run_parallel), l1_ratios))

        
        if len(np.unique(scores_df.stack()))==1:
            best_row, best_col = np.where(zeros_df.values == \
                                                  np.nanmax(zeros_df.values))
            best_l1 = zeros_df.index[np.nanmax(best_row)]
            best_C = zeros_df.columns[np.nanmin(best_col)]
        else:
            normed_scores = pd.DataFrame(self._min_max(scores_df.copy().values))
            normed_zeros = pd.DataFrame(self._min_max(zeros_df.copy().values))

            combination = 2 * ((normed_scores.copy().applymap(self._inv) + \
                           normed_zeros.copy().applymap(self._inv)
                           ).applymap(self._inv))

            combination.index = scores_df.index.copy()
            combination.columns = scores_df.columns.copy()

            best_combination_row, best_combination_col = np.where(combination == \
                                                      np.nanmax(combination.values))
            best_l1 = combination.index[np.nanmax(best_combination_row)]
            best_C = combination.columns[np.nanmin(best_combination_col)]
		
            self._combination_cv = combination
            self._combination_cv.columns.name = 'Harmonic Mean'

        return(best_C, best_l1)
    
    
    
    
    def _par_selection_BIC(self,
                        C,
                        l1_ratios):
        """
        Preselect best `C` and `l1 ratio` with cross-validation.
        PARAMETERS
        ----------
        C: <list of int or float values>
            List holding regularisation parameters for `K` models. The lower, the
            stronger the regularization is.
        l1_ratios: <list of int or float values>
            List holding ratios between l1 and l2 penalty. Must be in [0,1]. For
            pure l2 use 0, for pure l1 use 1.
        RETURNS
        -------
        <tuple>
            - First entry: suggested `C` parameter.
            - Second entry: suggested `l1 ratio`.
        """
        # self._AIC_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C) 
        self._BIC_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C) 
        def run_parallel(l1):
            """
            Parallel computation of for ``K`` * ``C`` * ``l1_ratios`` models.
            PARAMETERS
            -----
            l1: current l1 ratio in the parallelization framework.
            """
            for reg in C:
                if self._scale == True:
                    sc = StandardScaler()
                    train_data = sc.fit_transform(self._data)
                    train_target = self._target
                    
                elif self._scale == False:
                    train_data = self._data.values
                    train_target = self._target
                    

                sgd = LogisticRegression(penalty="elasticnet", C=reg,
                                            solver="saga", l1_ratio=l1,
                                            random_state=self._random_state)

                sgd.fit(train_data, train_target)

                num_params = len(np.where(sgd.coef_ != 0)[1]) + 1
                
                pred = sgd.predict_proba(train_data)
                log_likelihood = log_loss(y_true=train_target, y_pred=pred, normalize=False)
                
                # AIC = 2 * log_likelihood + 2 * num_params
                BIC =  2 * log_likelihood + np.log(len(train_target)) * num_params
                # self._AIC_df.loc[l1, reg] = AIC
                self._BIC_df.loc[l1, reg] = BIC
                
        Parallel(n_jobs=-1, verbose=1, backend="threading")(
             map(delayed(run_parallel), l1_ratios))

        
        best_combination_row, best_combination_col = np.where(self._BIC_df == \
                                                      np.nanmin(self._BIC_df.values))
        best_l1 = self._BIC_df.index[np.nanmax(best_combination_row)]
        best_C = self._BIC_df.columns[np.nanmin(best_combination_col)]

        return(best_C, best_l1)
    
    
    
    def run_parallel(self, K):
        """
        If ``autoEnetParSel=False``, parallel computation of ``K`` * ``len(C)`` \
            * ``len(l1_ratios)`` classification models. Otherwise, \
                computation of ``K`` models.     
        PARAMETERS
        ----------
        K: 
            Range of train-test splits. The parameter cannot be set directly \
                by the user but is used for an internal parallelization.
        """
        # Loop through all C
        for C in self._C:
            for l1 in self._l1_ratios:
                
                if self._random_state == None:
                    X_train, X_test, y_train, y_test = train_test_split(
                          self._data, self._target,
                          test_size=self._random_testsizes[K],
                          stratify=self._target, random_state=None)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                          self._data, self._target,
                          test_size=self._random_testsizes[K],
                          stratify=self._target, random_state=K)

                self._X_test = X_test

                if self._scale == True:
                    sc = StandardScaler()
                    sc.fit(X_train)
                    X_train_std = sc.transform(X_train)
                    X_test_std = sc.transform(X_test)
                elif self._scale == False:
                    X_train_std = X_train.copy().values
                    X_test_std = X_test.copy().values

                if self._verbose > 1:
                    print('C = ', C, 'l1 = ', l1, ', TT split = ', K)

                if self._classifier == 'logreg':
                    # Trian a logistic regreission model
                    model = LogisticRegression(solver='saga',
                                            C=C,
                                            penalty='elasticnet',
                                            l1_ratio=l1,
                                            n_jobs=-1,
                                            max_iter=5000,
                                            random_state=self._random_state).\
                                            fit(X_train_std, y_train)
                else:
                    sys.exit('No valid classifier.')

                # Get all weights (coefficients). Those that were selected
                # are non-zero, otherwise zero
                #print(logreg.coef_)
                self._weight_dict[(C, l1, K)] = model.coef_
                self._weight_list.append(model.coef_)

                if self._scoring == 'accuracy':
                    y_test_pred = model.predict(X_test_std)
                    score = model.score(X_test_std, y_test)
                elif self._scoring == 'f1':
                    y_test_pred = model.predict(X_test_std)
                    score = f1_score(y_test, y_test_pred)
                elif self._scoring == 'precision':
                    y_test_pred = model.predict(X_test_std)
                    score = precision_score(y_test, y_test_pred)
                elif self._scoring == 'recall':
                    y_test_pred = model.predict(X_test_std)
                    score = recall_score(y_test, y_test_pred)
                elif self._scoring == 'mcc':
                    y_test_pred = model.predict(X_test_std)
                    score = matthews_corrcoef(y_test, y_test_pred)

                #check if we need score_all and score_dict
                self._score_dict[(C, l1, K)] = score
                self._score_list.append(score)

                # Collect true values and predictions in dictionary
                predictions = pd.DataFrame({'y_test':y_test, \
                                       'y_pred': y_test_pred})
                predictions.index = X_test.index

                # calculate predict_proba for current train/test and weight
                # initialization
                self._predictions_dict[(C, l1, K)] = predictions
                if(self._classifier == 'logreg'):
                    self._probas[(C, l1, K)] = pd.DataFrame( \
                           model.predict_proba(X_test_std), index = \
                                X_test.index)
    def train(self):
        self._predictions_dict = {}
        self._probas = {}
        self._pred_proba_dict = {}
        super().train()

        # Build a dictionary with the prediction probabilities
        for C in self._C:
            for l1 in self._l1_ratios:
                count =  0
                vec = pd.DataFrame(np.nan, index= self._indices, \
                                columns = ['remove'])
                for k in self._probas.keys():

                    if k[0] == C and k[1] == l1:
                        vec.loc[self._probas[k].index,count] = \
                        self._probas[k].iloc[:, 1].values
                        count += 1
                vec = vec.iloc[:, 1:]
                self._pred_proba_dict[(C, l1)] = vec

    def get_summary_objects(self):
        """
        Each object of the dataset is a certain number between 
        0 (never) and ``K`` (always) part of the test set inside RENT training.
        This method computes a summary of classification results for each sample
        across all models, where the sample was part of the test set.
        The summary contains information on how often a sample has been mis-classfied.
        
        RETURNS
        -------
        <pandas dataframe>
            Data matrix. Rows represent objects, columns represent generated variables.
            The first column denotes how often the object was part of the test set, 
            the second column reveals the true class
            of the object, the third column indicates how often the object was 
            classified incorrectly and the fourth column shows the corresponding 
            percentage of incorrectness.
        """
        if not hasattr(self, '_best_C'):
            sys.exit('Run train() first!')

        self._incorrect_labels = pd.DataFrame({'# test':np.repeat\
                                      (0, np.shape(self._data)[0]),
                                      'class':self._target,
                                      '# incorrect':np.repeat\
                                      (0, np.shape(self._data)[0])})
        self._incorrect_labels.index=self._indices.copy()

        specific_predictions = []
        for K in range(self._K):
            specific_predictions.append(
                self._predictions_dict[(self._best_C, self._best_l1_ratio, K)])
        for dataframe in range(len(specific_predictions)):
            for count, tup in enumerate(
                    zip(specific_predictions[dataframe].y_test, \
                        specific_predictions[dataframe].y_pred)):
                ind = specific_predictions[dataframe].index[count]

                # Upgrade ind by one if used as test object
                self._incorrect_labels.loc[ind,'# test'] += 1
                if tup[0] != tup[1]:
                    # Upgrade number of incorrectly classified
                    self._incorrect_labels.loc[ind,'# incorrect'] += 1

        self._incorrect_labels['% incorrect'] = \
        (self._incorrect_labels["# incorrect"] \
             / self._incorrect_labels['# test']) * 100

        return self._incorrect_labels

    def get_object_probabilities(self):
        """
        Logistic Regression probabilities for each combination of object and model. 
        The method can only be used if ``classifier='logreg'``.
        
        RETURNS
        -------
        <pandas dataframe>
            Matrix, where rows represent objects and columns represent \
                logistic regression probability outputs (probability of \
                    belonging to class 1).
                    
        """

        if not hasattr(self, '_pred_proba_dict'):
            sys.exit('Run train() first!')

        # predicted probabilities only if Logreg
        if self._classifier != 'logreg':
            return warnings.warn('Classifier must be "logreg"!')
        else:
            self._pp_data = self._pred_proba_dict[
                (self._best_C, self._best_l1_ratio)].copy()

            self._pp_data.columns = ['mod {0}'.format(x+1) \
                                        for x in range(
                                                self._pp_data.shape[1])]
            return self._pp_data

    def plot_object_probabilities(self, object_id, binning='auto', lower=0,
                                  upper=1, kde=False, norm_hist=False):
        """
        Histograms of predicted probabilities from ``get_object_probabilities()``.
        
        PARAMETERS
        ----------
        object_id : <list of int or str>
            Objects whoes histograms shall be plotted. 
            Type depends on the index format of the dataframe.
        lower : <float>
            Lower bound of the x-axis. Default: ``lower=0``.
        upper : <float>
            Upper bound of the x-axis. Default: ``upper=1``.
        kde : <boolean>
            Kernel density estimation, from `seaborn distplot`.
            Default: ``kde=False``.
        norm_hist : <boolean>
            Normalize the histogram, from `seaborn distplot`.
            Default: ``norm_hist=False``.              
        """
        
        if not hasattr(self, '_best_C'):
            sys.exit('Run train() first!')
            
        target_objects = pd.DataFrame(self._target)
        target_objects.index = self._pred_proba_dict[self._best_C, \
                              self._best_l1_ratio].index
        self._t = target_objects
        for obj in object_id:
            fig, ax = plt.subplots()
            data = self._pred_proba_dict[self._best_C, \
                              self._best_l1_ratio].loc[obj,:].dropna()

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
            ax.axvline(x=0.5, color='k', linestyle='--', label ="Threshold")
            ax.legend(fontsize=10)

            if norm_hist == False:
                ax.set_ylabel('absolute frequencies', fontsize=10)
                ax.set_xlabel('ProbC1', fontsize=10)
            else:
                ax.set_ylabel('frequencies', fontsize=10)
                ax.set_xlabel('ProbC1', fontsize=10)
            ax.set_title('Object: {0}, True class: {1}'.format(obj, \
                         target_objects.loc[obj,:].values[0]), fontsize=10)
                
                
                

    def _prepare_validation_study(self, test_data, test_labels, num_drawings, 
                                  num_permutations, metric='mcc', alpha=0.05):

        # RENT prediction
        if self._scale == True:
            sc = StandardScaler()
            train_RENT = sc.fit_transform(self._data.iloc[:, self._sel_var])
            test_RENT = sc.transform(test_data.iloc[:, self._sel_var])
        elif self._scale == False:
            train_RENT = self._data.iloc[:, self._sel_var].values
            test_RENT = test_data.iloc[:, self._sel_var].values
        if self._classifier == 'logreg':
                    model = LogisticRegression(penalty='none', max_iter=8000,
                                                solver="saga", \
                                                random_state=self._random_state).\
                        fit(train_RENT,self._target)
        else:
            print("something")

        if metric == 'mcc':
                score = matthews_corrcoef(test_labels, model.predict(test_RENT))
        elif metric == 'f1':
            score = f1_score(test_labels, model.predict(test_RENT))
        elif metric == 'acc':
            score = accuracy_score(test_labels, model.predict(test_RENT))

        # VS1
        VS1 = []
        for K in range(num_drawings):
            # Randomly select features (# features = # RENT features selected)
            columns = np.random.RandomState(seed=K).choice(
                range(0,len(self._data.columns)),
                                    len(self._sel_var))
            if self._scale == True:
                sc = StandardScaler()
                train_VS1 = sc.fit_transform(self._data.iloc[:, columns])
                test_VS1 = sc.transform(test_data.iloc[:, columns])
            elif self._scale == False:
                train_VS1 = self._data.iloc[:, columns].values
                test_VS1 = test_data.iloc[:, columns].values
            
            if self._classifier == 'logreg':
                model = LogisticRegression(penalty='none', max_iter=8000,
                                            solver="saga", 
                                            random_state=self._random_state).\
                    fit(train_VS1,self._target)
            else:
                print("something")
            if metric == 'mcc':
                VS1.append(matthews_corrcoef(test_labels, \
                                                    model.predict(test_VS1)))
            elif metric == 'f1':
                VS1.append(f1_score(test_labels, \
                                                    model.predict(test_VS1)))
            elif metric == 'acc':
                VS1.append(accuracy_score(test_labels, \
                                                    model.predict(test_VS1)))
        
        # VS2
        sc = StandardScaler()
        test_data.columns = self._data.columns
        VS2 = []
        if self._scale == True:
            train_VS2 = sc.fit_transform(self._data.iloc[:,self._sel_var])
            test_VS2 = sc.transform(test_data.iloc[:, self._sel_var])
        elif self._scale == False:
            train_VS2 = self._data.iloc[:,self._sel_var].values
            test_VS2 = test_data.iloc[:, self._sel_var].values

        if self._classifier == 'logreg':
            model = LogisticRegression(penalty='none', max_iter=8000,
                                        solver="saga", 
                                        random_state=self._random_state ).\
                    fit(train_VS2, self._target)
        else:
            print("add model")

        for K in range(num_permutations):
            if metric == 'mcc':
                VS2.append(matthews_corrcoef(
                        np.random.RandomState(seed=K).permutation(test_labels),\
                        model.predict(test_VS2)))
            elif metric == 'f1':
                VS2.append(f1_score(
                        np.random.RandomState(seed=K).permutation(test_labels),\
                        model.predict(test_VS2)))
            elif metric == 'acc':
                VS2.append(accuracy_score(
                        np.random.RandomState(seed=K).permutation(test_labels),\
                        model.predict(test_VS2)))
                
        return score, VS1, VS2
                    
                    
        
class RENT_Regression(RENT_Base):
    """
    This class carries out RENT on a given regression dataset. 
    
    PARAMETERS
    ----------
    
    data: <numpy array> or <pandas dataframe>
        Dataset on which feature selection shall be performed. \
            Variable types must be numeric or integer.            
    target: <numpy array> or <pandas dataframe>
        Response variable of data.        
    feat_names : <list>
        List holding feature names. Preferably a list of string values. \
            If empty, feature names will be generated automatically. \
                Default: ``feat_names=[]``.
    C : <list of int or float values>
        List with regularisation parameters for ``K`` models. The lower,
        the stronger the regularization is. Default: ``C=[1,10]``.        
    l1_ratios : <list of int or float values>
        List holding ratios between l1 and l2 penalty. Values must be in [0,1]. For
        pure l2 use 0, for pure l1 use 1. Default: ``l1_ratios=[0.6]``.        
    autoEnetParSel : <boolean>
        Cross-validated elastic net hyperparameter selection.
            - ``autoEnetParSel=True`` : peform a cross-validation \
                pre-hyperparameter search, such that RENT runs only with \
                    one hyperparamter setting.
            - ``autoEnetParSel=False`` : perform RENT with each combination of \
                ``C`` and ``l1_ratios``. Default: ``autoEnetParSel=True``.
    BIC : <boolean>
        Use the Bayesian information criterion to select hyperparameters.
            - ``BIC=True`` : use BIC to select RENT hyperparameters.
            - ``BIC=False``: no use of BIC. 
    poly : <str> 
        Create non-linear features. Default: ``poly='OFF'``.
            - ``poly='OFF'`` : no feature interaction.
            - ``poly='ON'`` : feature interaction and squared features (2-polynoms).
            - ``poly='ON_only_interactions'`` : only feature interactions, \
                no squared features.                
    testsize_range : <tuple float>
         Inside RENT, ``K`` models are trained, where the testsize defines the \
             proportion of train data used for testing of a single model. The testsize 
         can either be randomly selected inside the range of ``testsize_range`` for \
             each model or fixed by setting the two tuple entries to the same value. 
         The tuple must be in range (0,1).
         Default: ``testsize_range=(0.2, 0.6)``.         
    K : <int>
        Number of unique train-test splits. Default ``K=100``.   
    scale : <boolean>
        Columnwise standardization of the K train datasets. Default ``scale=True``.    
    random_state : <None or int>
        Set a random state to reproduce your results. Default: ``random_state=None``.
            - ``random_state=None`` : no random seed. 
            - ``random_state={0,1,2,...}`` : random seed set.
    verbose : <int>
        Track the train process if value > 1. If ``verbose = 1``, only the overview
        of RENT input will be shown. Default: ``verbose=0``.
        
    RETURNS
    ------
    <class>
        A class that contains the RENT regression model.
    """
    __slots__ =["_data", "_target", "_feat_names", "_C", "_l1_ratios", "_autoEnetParSel",
               "_BIC", "_poly", "_testsize_range", "_K", "_scale", "_random_state",
               "_verbose", "_summary_df", "_score_dict", "_BIC_df", "_best_C",
               "_best_l1_ratio", "_indices", "_runtime", "_scores_df", "_combination", 
               "_zeros", "_perc", "_self_var", "_scores_df_cv", "_zeros_df_cv", "_combination_cv", 
               "_predictions_abs_errors", "_random_testsizes", "_weight_dict", "_weight_list", 
               "_score_list", "_histogram_data"]


    def __init__(self, data, target, feat_names=[], 
                 C=[1,10], l1_ratios = [0.6], autoEnetParSel=True, BIC=False,
                 poly='OFF', testsize_range=(0.2, 0.6),
                 K=100, scale=True, random_state = None, verbose = 0):


        super().__init__(data, target, feat_names, C, l1_ratios, 
                         autoEnetParSel, BIC, poly, testsize_range, K, scale, 
                         random_state, verbose)

    def _par_selection(self,
                    C,
                    l1_ratios,
                    n_splits=5,
                    testsize_range=(0.25,0.25)):
        """
        Preselect best `C` and `l1 ratio` with Cross-validation.
        
        PARAMETERS
        ----------
        C: <list of int or float values>
        List holding regularisation parameters for `K` models. The lower, the
        stronger the regularization is.        
        l1_ratios: <list of int or float values>
            List holding ratios between l1 and l2 penalty. Must be in [0,1]. For
            pure l2 use 0, for pure l1 use 1.
        n_splits : <int>
            Number of cross-validation folds. Default ``n_splits=5``.
        testsize_range: <tuple float>
            Range of random proportion of dataset to include in test set,
            low and high are floats between 0 and 1. Default ``testsize_range=(0.2, 0.6)``.
            Testsize can be fixed by setting low and high to the same value.
            
        RETURNS
        -------
        <tuple> 
            First entry: suggested `C` parameter.
            Second entry: suggested `l1 ratio`.
            
        """
        skf = KFold(n_splits=n_splits, random_state=self._random_state, shuffle=True)
        scores_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C)
        zeros_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C)

        def run_parallel(l1):
            """
            Parallel computation of for ``K`` * ``C`` * ``l1_ratios`` models.
            
            PARAMETERS
            ----------
            l1: current l1 ratio in the parallelization framework.
            """
            for reg in C:
                scores = list()
                zeros = list()
                for train, test in skf.split(self._data, self._target):
                    # Find those parameters that are 0
                    if self._scale == True:
                        sc = StandardScaler()
                        train_data = sc.fit_transform(self._data.iloc[train,:])
                        train_target = self._target[train]
                        test_data_split = sc.transform(self._data.iloc[test,:])
                        test_target = self._target[test]
                    elif self._scale == False:
                        train_data = self._data.iloc[train,:].values
                        train_target = self._target[train]
                        test_data_split = self._data.iloc[test,:].values
                        test_target = self._target[test]

                    sgd =  ElasticNet(alpha=1/reg, l1_ratio=l1,
                                       max_iter=5000, 
                                       random_state=self._random_state, \
                                       fit_intercept=False).\
                                       fit(train_data, train_target)

                    mod_coef = sgd.coef_.reshape(1, len(sgd.coef_))
                    params = np.where(mod_coef != 0)[1]

                    # if there are parameters != 0, build a predicion model and
                    # find best parameter combination w.r.t. scoring
                    if len(params) == 0:
                        scores.append(np.nan)
                        zeros.append(np.nan)
                    else:
                        zeros.append((len(self._data.columns)-len(params))\
                                      /len(self._data.columns))

                        train_data_1 = train_data[:,params]
                        test_data_1 = test_data_split[:, params]

                        model = LinearRegression().\
                                fit(train_data_1, train_target)
                        scores.append(r2_score(test_target, \
                                        model.predict(test_data_1)))

                scores_df.loc[l1, reg] = np.nanmean(scores)
                zeros_df.loc[l1, reg] = np.nanmean(zeros)

        Parallel(n_jobs=-1, verbose=0, backend="threading")(
             map(delayed(run_parallel), l1_ratios))

        s_arr = scores_df.stack()
        if len(np.unique(s_arr))==1:
            best_row, best_col = np.where(zeros_df.values == \
                                                  np.nanmax(zeros_df.values))
            best_l1 = zeros_df.index[np.nanmax(best_row)]
            best_C = zeros_df.columns[np.nanmin(best_col)]
        else:
            normed_scores = pd.DataFrame(self._min_max(scores_df.values))
            normed_zeros = pd.DataFrame(self._min_max(zeros_df.values))

            combination = 2 * ((normed_scores.copy().applymap(self._inv) + \
                           normed_zeros.copy().applymap(self._inv)
                           ).applymap(self._inv))
            combination.index = scores_df.index.copy()
            combination.columns = scores_df.columns.copy()
            best_combination_row, best_combination_col = np.where(combination == \
                                                      np.nanmax(combination.values))
                                               
            best_l1 = combination.index[np.nanmax(best_combination_row)]
            best_C = combination.columns[np.nanmin(best_combination_col)]

        self._scores_df_cv, self._zeros_df_cv, self._combination_cv = \
            scores_df, zeros_df, combination

        self._scores_df_cv.columns.name = 'Scores'
        self._zeros_df_cv.columns.name = 'Zeros'
        self._combination_cv.columns.name = 'Harmonic Mean'
        return(best_C, best_l1)
    
    
    def _par_selection_BIC(self,
                        C,
                        l1_ratios):
        """
        Preselect best `C` and `l1 ratio` with the Bayesian information criterion.
        PARAMETERS
        ----------
        C: <list of int or float values>
            List holding regularisation parameters for `K` models. The lower, the
            stronger the regularization is.
        l1_ratios: <list of int or float values>
            List holding ratios between l1 and l2 penalty. Must be in [0,1]. For
            pure l2 use 0, for pure l1 use 1.
        RETURNS
        -------
        <tuple>
            - First entry: suggested `C` parameter.
            - Second entry: suggested `l1 ratio`.
        """
        # self._AIC_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C) 
        self._BIC_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C) 
        def run_parallel(l1):
            """
            Parallel computation of for ``K`` * ``C`` * ``l1_ratios`` models.
            PARAMETERS
            -----
            l1: current l1 ratio in the parallelization framework.
            """
            for reg in C:
                if self._scale == True:
                    sc = StandardScaler()
                    train_data = sc.fit_transform(self._data)
                    train_target = self._target
                    
                elif self._scale == False:
                    train_data = self._data.values
                    train_target = self._target
                    

                sgd =  ElasticNet(alpha=1/reg, l1_ratio=l1,
                                       max_iter=5000, 
                                       random_state=self._random_state, \
                                       fit_intercept=False).\
                                       fit(train_data, train_target)

                mod_coef = sgd.coef_.reshape(1, len(sgd.coef_))
                #params = np.where(mod_coef != 0)[1]
                self.test_coef = mod_coef
                num_params = len(np.where(mod_coef != 0)[1]) + 1
                
                pred = sgd.predict(train_data)
                #log_likelihood = log_loss(y_true=train_target, y_pred=pred, normalize=False)
                
                sigma_2 = np.var(self._target, ddof=1)
                SSE = np.sum((pred - self._target)**2)
                n = len(pred)
                
                # AIC = n * np.log(2*np.pi *sigma_2) + 1/sigma_2 * SSE + 2*num_params
                # self._AIC_df.loc[l1, reg] = AIC
                self._BIC_df.loc[l1,reg] = n * np.log(2*np.pi *sigma_2) + 1/sigma_2 * SSE + np.log(n) * num_params 
                
        Parallel(n_jobs=-1, verbose=1, backend="threading")(
             map(delayed(run_parallel), l1_ratios))

        
        best_combination_row, best_combination_col = np.where(self._BIC_df == \
                                                      np.nanmin(self._BIC_df.values))
        best_l1 = self._BIC_df.index[np.nanmax(best_combination_row)]
        best_C = self._BIC_df.columns[np.nanmin(best_combination_col)]

        return(best_C, best_l1)
    

    def run_parallel(self, K):
        """
        If ``autoEnetParSel=False``, parallel computation of ``K`` * ``len(C)`` * \
            ``len(l1_ratios)`` linear regression models. Otherwise, \
                computation of ``K`` models.
        
        PARAMETERS
        -----
        K: 
            Range of train-test splits. The parameter cannot be set directly \
                by the user but is used for an internal parallelization.
        """

        # Loop through all C
        for C in self._C:
            # Loop through requested number of tt splits
            for l1 in self._l1_ratios:
                
                if self._random_state == None:
                    X_train, X_test, y_train, y_test = train_test_split(
                              self._data, self._target,
                              test_size=self._random_testsizes[K],
                              random_state=None)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                              self._data, self._target,
                              test_size=self._random_testsizes[K],
                              random_state=K)

                self._X_test = X_test

                if self._scale == True:
                    sc = StandardScaler()
                    sc.fit(X_train)
                    X_train_std = sc.transform(X_train)
                    X_test_std = sc.transform(X_test)
                if self._scale == False:
                    X_train_std = X_train.copy().values
                    X_test_std = X_test.copy().values

                if self._verbose > 1:
                    print('l1 = ', l1, 'C = ', C, ', TT split = ', K)

                model = ElasticNet(alpha=1/C, l1_ratio=l1,
                                       max_iter=5000, random_state=self._random_state, \
                                       fit_intercept=False).\
                                       fit(X_train_std, y_train)

                # Get all weights (coefficients). Those that were selected
                # are non-zero, otherwise zero
                mod_coef = model.coef_.reshape(1, len(model.coef_))
                self._weight_dict[(C, l1, K)] = mod_coef
                self._weight_list.append(mod_coef)

                pred = model.predict(X_test_std)
                abs_error_df = pd.DataFrame({'abs error': abs(y_test-pred)})
                abs_error_df.index = X_test.index
                self._predictions_abs_errors[(C, l1, K)] = abs_error_df

                score = r2_score(y_test,pred)
                self._score_dict[(C, l1, K)] = score
                self._score_list.append(score)
    
    def train(self):
        self._predictions_abs_errors = {}
        super().train()

    def get_summary_objects(self):
        """
        Each object of the dataset is a certain number between 0 (never) and ``K`` 
        (always) part of th test set inside RENT training.
        This method computes a summary of the mean absolute errors for each sample
        across all models, where the sample was part of the test set.
        
        Returns
        -------
        <pandas dataframe>
            Data matrix. Rows represent objects, columns represent generated variables. 
            The first column denotes how often the object was part of the test set, \
                the second column shows the average absolute error.
                
        """
        if not hasattr(self, '_best_C'):
            sys.exit('Run train() first!')

        self._incorrect_labels = pd.DataFrame({'# test':np.repeat\
                                      (0, np.shape(self._data)[0]),
                                      'mean abs error':np.repeat\
                                      (0, np.shape(self._data)[0])})
        self._incorrect_labels.index=self._indices.copy()

        specific_predictions = []
        for K in range(self._K):
            specific_predictions.append(self._predictions_abs_errors[
                (self._best_C, self._best_l1_ratio, K)])
        self._histogram_data = pd.concat(specific_predictions, axis=1)
        self._histogram_data.columns = ['mod {0}'.format(x+1) \
                                        for x in range(
                                                self._histogram_data.shape[1])]
        count = self._histogram_data.count(axis=1)
        mean_abs_error = np.nanmean(self._histogram_data, axis=1)

        summary_df = pd.DataFrame({'count': count, 'mae': mean_abs_error})
        summary_df.index = self._histogram_data.index

        self._incorrect_labels.loc[summary_df.index,'# test'] = \
        summary_df.loc[:,'count']
        self._incorrect_labels.loc[summary_df.index,'mean abs error'] = \
        summary_df.loc[:,'mae']

        self._incorrect_labels.iloc[ \
            np.where(self._incorrect_labels.iloc[:,0] == 0)[0]] = np.nan

        return self._incorrect_labels

    def get_object_errors(self):
        """
        Absolute errors for samples which were at least once in a test-set among ``K``
        models.
        
        Returns
        -------
        <pandas dataframe>
            Matrix. Rows represent objects, columns represent genrated variables.
            
        """
        if not hasattr(self, '_histogram_data'):
            sys.exit('Run get_summary_objects() first!')
        return self._histogram_data

    def plot_object_errors(self, object_id, binning='auto', lower=0,
                                  upper=100, kde=False, norm_hist=False):
        """
        Histograms of absolute errors from ``get_object_errors()``.
        
        PARAMETERS
        ----------
        object_id : <list of int or str>
            Objects whoes histograms shall be plotted. Type depends \
                on the index format of the dataframe.
        lower : <float>
            Lower bound of the x-axis. Default ``lower=0``.
        upper : <float>
            Upper bound of the x-axis. Default ``upper=100``.
        kde : <boolean>
            Kernel density estimation, from `seaborn distplot`.
            Default: ``kde=False``.
        norm_hist : <boolean>
            Normalize the histogram, from `seaborn distplot`.
            Default: ``norm_hist=False``.
        """
        if not hasattr(self, '_histogram_data'):
            sys.exit('Run get_summary_objects() first!')
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
                ax.set_ylabel('frequencies', fontsize=10)
                ax.set_xlabel('Absolute Error', fontsize=10)
            ax.set_title('Object: {0}'.format(obj), fontsize=10)
    
    
    def _prepare_validation_study(self, test_data, test_labels, num_drawings, 
                                  num_permutations, metric=None, alpha=0.05):
        
        # RENT prediction
        if self._scale == True:
            sc = StandardScaler()
            train_RENT = sc.fit_transform(self._data.iloc[:, self._sel_var])
            test_RENT = sc.transform(test_data.iloc[:, self._sel_var])
        elif self._scale == False:
            train_RENT = self._data.iloc[:, self._sel_var].values
            test_RENT = test_data.iloc[:, self._sel_var].values
        model = LinearRegression().fit(train_RENT,self._target)
        score = r2_score(test_labels, model.predict(test_RENT))

        # VS1
        VS1 = []
        for K in range(num_drawings):
            # Randomly select features (# features = # RENT features selected)
            columns = np.random.RandomState(seed=K).choice(
                range(len(self._data.columns)), len(self._sel_var)
            )

            if self._scale == True:
                sc = StandardScaler()
                train_VS1 = sc.fit_transform(self._data.iloc[:, columns])
                test_VS1 = sc.transform(test_data.iloc[:, columns])
            elif self._scale == False:
                train_VS1 = self._data.iloc[:, columns].values
                test_VS1 = test_data.iloc[:, columns].values

            model = LinearRegression().fit(train_VS1, self._target)
            VS1.append(r2_score(test_labels, model.predict(test_VS1)))

        # VS2
        sc = StandardScaler()
        test_data.columns = self._data.columns
        if self._scale == True:
            train_VS2 = sc.fit_transform(self._data.iloc[:,self._sel_var])
            test_VS2 = sc.transform(test_data.iloc[:, self._sel_var])
        elif self._scale == False:
            train_VS2 = self._data.iloc[:,self._sel_var].values
            test_VS2 = test_data.iloc[:, self._sel_var].values

        model = LinearRegression().fit(train_VS2,self._target)
        VS2 = [r2_score(
            np.random.RandomState(seed=K).permutation(test_labels),\
            model.predict(test_VS2)) for K in range(num_permutations)]
        return score, VS1, VS2
    
