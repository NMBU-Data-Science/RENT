Quickstart
==========

RENT (Repeated Elastic Net Technique) is a package for feature selection for binary classification problems and regression problems. At its core
RENT trains an ensemble of unique models using regularized elastic net to select features. Each model in the ensemble is trained with
a unique and randomly selected subset from the full training data. From these models one can acquire weight distributions for each
feature that contain rich information on the stability of feature selection and from which several adjustable classification criteria may be
defined.

More details are provided here: [RENT - Repeated Elastic Net Technique for Feature Selection](https://arxiv.org/abs/2009.12780)

Requirements
------------

some info here


Documentation
-------------
The following Jupyter notebooks provides a [classification example](https://github.com/NMBU-Data-Science/RENT/blob/master/src/RENT/Classification_example.ipynb) and a [regression example](https://github.com/NMBU-Data-Science/RENT/blob/master/src/RENT/Regression_example.ipynb), illustrating the RENT workflow.


RENT repository on GitHub
----------------------------
The source code is available at the `RENT GitHub repository`_.

.. _RENT GitHub repository: https://github.com/NMBU-Data-Science/RENT


Testing
-------

To be implemented.
