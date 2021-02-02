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
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the `Anaconda distribution`_.

.. _Anaconda distribution: https://www.anaconda.com/products/individual

    - numpy >= 1.11.3
    - pandas >= 1.0.5
    - scikit-learn >= 0.22
    - scipy >= 1.5.0
    - hoggorm >= 0.13.3
    - hoggormplot >= 0.13.2
    - matplotlib >= 3.2.2
    - seaborn >= 0.10



Documentation
-------------
The following Jupyter notebooks provides a `classification example <https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Classification_example.ipynb>`_ and a `regression example <https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Regression_example.ipynb>`_, illustrating the RENT workflow.


RENT repository on GitHub
----------------------------
The source code is available at the `RENT GitHub repository`_.

.. _RENT GitHub repository: https://github.com/NMBU-Data-Science/RENT


Testing
-------

To be implemented.
