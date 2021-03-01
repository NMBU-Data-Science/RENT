RENT
====

RENT (Repeated Elastic Net Technique) is a feature selection method for binary classification and regression problems. At its core
RENT trains an ensemble of unique models using regularized elastic net to select features. Each model in the ensemble is trained with
a unique and randomly selected subset from the full training data. From these models one can acquire weight distributions for each
feature that contain rich information on the stability of feature selection and from which several adjustable classification criteria may be
defined.

More details are provided here: [RENT - Repeated Elastic Net Technique for Feature Selection](https://arxiv.org/abs/2009.12780v2)

# Example

We provide an example Jupyter-notebook for both [classification](https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Classification_example.ipynb) and [regression](https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Regression_example.ipynb) on how to use RENT.

More example Jupyter notebooks are to come soon.


# Requirements

numpy >= 1.11.3   
pandas >= 1.0.5   
scikit-learn >= 0.22   
scipy >= 1.5.0  
hoggorm >= 0.13.3


# Documentation

A webhook to ReadTheDocs will be established soon. Until then, please see the docstrings inside the code. There you will find information on input parameters, etc.

Documentation at [ReadTheDocs](https://rent.readthedocs.io/en/latest/)

