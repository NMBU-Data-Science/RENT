import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from RENT import RENT



def test_summary_criteria():
    expected_results = pd.read_csv('tests/csv/summary_criteria.csv')

    # load the data
    wisconsin = load_breast_cancer()
    data = pd.DataFrame(wisconsin.data)
    data.columns = wisconsin.feature_names
    target = wisconsin.target

    # split
    train_data, test_data, train_labels, test_labels = train_test_split(data, target, random_state=0, shuffle=True)


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
                                        verbose=1)

    analysis.train()
    selected_features = analysis.selectFeatures(tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975)
    results = analysis.summary_criteria().reset_index()

    assert np.all(results.columns == expected_results.columns)
    columns = results.columns
    assert np.allclose(results[columns[1:]].values, expected_results[columns[1:]].values, equal_nan=True)
    assert results[columns[0]].equals(expected_results[columns[0]])
