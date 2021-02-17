import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from RENT import RENT



def test_RENT_Regression():
    # Summary criteria
    expected_results_summary_criteria = pd.read_csv('tests/csv_regression/summary_criteria.csv')
    # Summary objects
    expected_results_summary_objects = pd.read_csv('tests/csv_regression/summary_objects.csv')
    # Object probabilities
    expected_results_object_errors = pd.read_csv('tests/csv_regression/object_errors.csv')
    # Weights
    expected_results_weights = pd.read_csv('tests/csv_regression/weights.csv')

    # Load the data
    data = make_regression(n_samples=250, n_features=1000, n_informative=20, random_state=0, shuffle=False)
    my_data = pd.DataFrame(data[0])
    my_target = data[1]
    my_feat_names = ['f{0}'.format(x+1) for x in range(len(my_data.columns))]

    #split
    train_data, test_data, train_labels, test_labels = train_test_split(my_data, my_target, test_size=0.3, random_state=0)


    # Define a range of regularisation parameters C for elastic net. A minimum of at least one value is required.
    my_C_params = [0.1, 1, 10]
    # Define a reange of l1-ratios for elastic net.  A minimum of at least one value is required.
    my_l1_ratios = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
    # Define setting for RENT
    analysis = RENT.RENT_Regression(data=train_data, 
                                    target=train_labels, 
                                    feat_names=my_feat_names, 
                                    C= my_C_params, 
                                    l1_ratios=my_l1_ratios,
                                    autoEnetParSel=True,
                                    poly='OFF',
                                    testsize_range=(0.25,0.25),
                                    K=100,
                                    random_state=0,
                                    verbose=0)

    analysis.train()
    selected_features = analysis.selectFeatures(tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975)
    summary_criteria = analysis.summary_criteria().reset_index()
    summary_objects = analysis.summary_objects().reset_index()
    object_errors = analysis.get_object_errors().reset_index()
    weights = analysis.get_weight_distributions().reset_index()

    # Summary criteria 
    assert np.all(summary_criteria.columns == expected_results_summary_criteria.columns)
    columns = summary_criteria.columns
    assert np.allclose(summary_criteria[columns[1:]].values, expected_results_summary_criteria[columns[1:]].values, equal_nan=True)
    assert summary_criteria[columns[0]].equals(expected_results_summary_criteria[columns[0]])

    # Summary objects
    assert np.all(summary_objects.columns == expected_results_summary_objects.columns)
    columns = summary_objects.columns
    assert np.allclose(summary_objects[columns[1:]].values, expected_results_summary_objects[columns[1:]].values, equal_nan=True)
    assert summary_objects[columns[0]].equals(expected_results_summary_objects[columns[0]])

    # Object probabilites
    assert np.all(object_errors.columns == expected_results_object_errors.columns)
    columns = object_errors.columns
    #sort by rows because the single probabilities can be on different places due to paralellization
    df1 = object_errors[columns[1:]].copy().transpose()
    df2 = expected_results_object_errors[columns[1:]].copy().transpose()
    for col in df1:
        df1[col] = df1[col].sort_values(ignore_index=True)
    for col in df2:
        df2[col] = df2[col].sort_values(ignore_index=True)
    assert np.allclose(df1.values, df2.values, equal_nan=True)
    assert object_errors[columns[0]].equals(expected_results_object_errors[columns[0]])

    # Weights
    assert np.all(weights.columns == expected_results_weights.columns)
    columns = weights.columns
    # sort by rows because the single probabilities can be on different places due to paralellization
    df1 = weights[columns[1:]].copy().transpose()
    df2 = expected_results_weights[columns[1:]].copy().transpose()
    for col in df1:
        df1[col] = df1[col].sort_values(ignore_index=True)
    for col in df2:
        df2[col] = df2[col].sort_values(ignore_index=True)
    assert np.allclose(df1.values, df2.values, equal_nan=True)
    assert weights[columns[0]].equals(expected_results_weights[columns[0]])
