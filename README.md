RENT
====

<img src="/images/RENT_logo.png" width="200"/>

RENT (Repeated Elastic Net Technique) is a feature selection method for binary classification and regression problems. At its core
RENT trains an ensemble of unique models using regularized elastic net to select features. Each model in the ensemble is trained with
a unique and randomly selected subset from the full training data. From these models one can acquire weight distributions for each
feature that contain rich information on the stability of feature selection and from which several adjustable classification criteria may be
defined.

More details are provided here: [RENT - Repeated Elastic Net Technique for Feature Selection](https://arxiv.org/abs/2009.12780v2)

Example
-------

Below are links to Jupyter-notebooks that illustrate how to use RENT for	

* [classification](https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Classification_example.ipynb) 
* [regression](https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Regression_example.ipynb)
* [hyperparameter search](https://github.com/NMBU-Data-Science/RENT/blob/master/examples/Extensive_hyperparameter_search.ipynb)



Requirements
------------
Make sure that Python 3.5 or higher is installed. A convenient way to install Python and many useful packages for scientific computing is to use the [Anaconda Distribution](https://www.anaconda.com/products/individual)

* numpy >= 1.11.3
* pandas >= 1.2.3
* scikit-learn >= 0.22
* scipy >= 1.5.0
* hoggorm >= 0.13.3
* hoggormplot >= 0.13.2
* matplotlib >= 3.2.2
* seaborn >= 0.10



Installation
------------
To install the package with the pip package manager, run the following command:  
`python3 -m pip install git+https://github.com/NMBU-Data-Science/RENT.git`



Documentation
-------------

Documentation is available at [ReadTheDocs](https://rent.readthedocs.io/en/latest/). It provides detailed explanation of methods and their inputs.

