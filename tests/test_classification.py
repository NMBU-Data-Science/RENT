import sys
sys.path.append('../src')
from RENT import RENT

import pandas as pd
import numpy as np


# Summary criteria
expected_results_summary_criteria = pd.read_csv('../tests/csv_classification/summary_criteria.csv')

# Summary objects
expected_results_summary_objects = pd.read_csv('../tests/csv_classification/summary_objects.csv')

# Object probabilities
expected_results_object_probabilities = pd.read_csv('../tests/csv_classification/object_probabilities.csv')

# Weights
expected_results_weights = pd.read_csv('../tests/csv_classification/weights.csv')

# load the data
train_data = pd.read_csv("../examples/data/wisconsin_train.csv").iloc[:, 1:]
train_labels = pd.read_csv("../examples/data/wisconsin_train_labels.csv").iloc[:, 1].values

# Define a range of regularisation parameters C for elastic net. A minimum of at least one value is required.
my_C_params = [0.1, 1, 10]

# Define a reange of l1-ratios for elastic net.  A minimum of at least one value is required.
my_l1_ratios = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]


# Define setting for RENT
analysis = RENT.RENT_Classification(data=train_data,
                                    target=train_labels,
                                    feat_names=train_data.columns,
                                    C=my_C_params,
                                    l1_ratios=my_l1_ratios,
                                    autoEnetParSel=True,
                                    poly='OFF',
                                    testsize_range=(0.25,0.25),
                                    scoring='mcc',
                                    classifier='logreg',
                                    K=100,
                                    random_state = 0,
                                    verbose=0)

analysis.train()
selected_features = analysis.select_features(tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975)
summary_criteria = analysis.get_summary_criteria().reset_index()
summary_objects = analysis.get_summary_objects().reset_index()
object_probabilities = analysis.get_object_probabilities().reset_index()
weights = analysis.get_weight_distributions().reset_index()


def test_summary_criteria_columns():
    """
    Verify column names of summary criteria dataframe.
    """
    assert np.all(summary_criteria.columns == expected_results_summary_criteria.columns)


def test_summary_criteria_values():
    """
    Verify values of summary criteria dataframe.
    """
    columns = summary_criteria.columns
    assert np.allclose(summary_criteria[columns[1:]].values, expected_results_summary_criteria[columns[1:]].values, equal_nan=True)


def test_summary_criteria_index():
    """
    Verify index or row names of summary criteria dataframe.
    """
    columns = summary_criteria.columns
    assert summary_criteria[columns[0]].equals(expected_results_summary_criteria[columns[0]])


def test_summary_objects_columns():
    """
    Verify column names of summary objects dataframe.
    """
    assert np.all(summary_objects.columns == expected_results_summary_objects.columns)


def test_summary_objects_values():
    """
    Verify values of summary objects.
    """
    columns = summary_objects.columns
    assert np.allclose(summary_objects[columns[1:]].values, expected_results_summary_objects[columns[1:]].values, equal_nan=True)


def test_summary_objects_index():
    """
    Verify index or row names of summary criteria dataframe.
    """
    columns = summary_objects.columns
    assert summary_objects[columns[0]].equals(expected_results_summary_objects[columns[0]])


def test_object_probabilities_columns():
    """
    Verify column names of object probabilities dataframe. 
    """
    assert np.all(object_probabilities.columns == expected_results_object_probabilities.columns)


def test_object_probabilities_values():
    """
    Verify values of object probabilities dataframe. 
    """
    columns = object_probabilities.columns
    
    # sort by rows because the single probabilities can be on different places due to paralellization
    df1 = object_probabilities[columns[1:]].copy().transpose()
    df2 = expected_results_object_probabilities[columns[1:]].copy().transpose()
    
    for col in df1:
        df1[col] = df1[col].sort_values(ignore_index=True)
    for col in df2:
        df2[col] = df2[col].sort_values(ignore_index=True)
    
    assert np.allclose(df1.values, df2.values, equal_nan=True)


def test_object_probabilities_index():
    """
    Verify values of object probabilities dataframe. 
    """
    columns = object_probabilities.columns
    
    # sort by rows because the single probabilities can be on different places due to paralellization
    df1 = object_probabilities[columns[1:]].copy().transpose()
    df2 = expected_results_object_probabilities[columns[1:]].copy().transpose()
    
    for col in df1:
        df1[col] = df1[col].sort_values(ignore_index=True)
    for col in df2:
        df2[col] = df2[col].sort_values(ignore_index=True)
    
    assert object_probabilities[columns[0]].equals(expected_results_object_probabilities[columns[0]])


def test_weights_columns():
    """
    Verify columns of weights dataframe.
    """
    assert np.all(weights.columns == expected_results_weights.columns)


def test_weights_values():
    """
    Verify values of weights dataframe. 
    """
    columns = weights.columns
    
    # sort by rows because the weights can be on different places due to paralellization
    df1 = weights[columns[1:]].copy().transpose()
    df2 = expected_results_weights[columns[1:]].copy().transpose()
    for col in df1:
        df1[col] = df1[col].sort_values(ignore_index=True)
    for col in df2:
        df2[col] = df2[col].sort_values(ignore_index=True)
    
    assert np.allclose(df1.values, df2.values, equal_nan=True)
    

def test_weights_index():
    """
    Verify index of weights dataframe. 
    """
    columns = weights.columns
    
    # sort by rows because the weights can be on different places due to paralellization
    df1 = weights[columns[1:]].copy().transpose()
    df2 = expected_results_weights[columns[1:]].copy().transpose()
    for col in df1:
        df1[col] = df1[col].sort_values(ignore_index=True)
    for col in df2:
        df2[col] = df2[col].sort_values(ignore_index=True)
    
    assert weights[columns[0]].equals(expected_results_weights[columns[0]])



