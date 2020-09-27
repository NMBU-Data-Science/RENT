RENT
====

RENT (Repeated Elastic Net Technique) is a feature selection method for binary classification and regression problems. At its core
RENT trains an ensemble of unique models using regularized elastic net to select features. Each model in the ensemble is trained with
a unique and randomly selected subset from the full training data. From these models one can acquire weight distributions for each
feature that contain rich information on the stability of feature selection and from which several adjustable classification criteria may be
defined.

# Example

The following Jupyter notebook provides and [example](https://github.com/NMBU-Data-Science/RENT/blob/master/src/RENT/Example.ipynb) of how to use RENT.


# Requirements

numpy >= 1.11.3   
pandas >= 1.0.5   
scikit-learn >= 0.22   
scipy >= 1.5.0   
hoggorm >= 0.13.3


# Documentation

A webhook to ReadTheDocs will be established soon. Until then, please see the docstrings inside the code. There you will find information on input parameters, etc.

Documentation at ReadTheDocs (no webhook yet): https://rent.readthedocs.io/en/latest/

