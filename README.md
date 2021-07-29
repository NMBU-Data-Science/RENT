RENT
====

<img src="/images/RENT_logo.png" width="200"/>

RENT (Repeated Elastic Net Technique) is a feature selection method for binary classification and regression problems. At its core
RENT trains an ensemble of <img src="https://render.githubusercontent.com/render/math?math=K\in\mathbb{N}"> generalized linear models using regularized elastic net to select features. Each model <img src="https://render.githubusercontent.com/render/math?math=k=1:K"> in the ensemble is trained using a randomly, iid sampled subset of rows of the full training data. A single data point can appear at most once in each subset, but may appear in multiple subsets. From these <img src="https://render.githubusercontent.com/render/math?math=K">unique models one can acquire weight distributions for each
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


Citing RENT
--------------

If you use RENT in a report or scientific publication, we would appreciate citations to the following paper:

[![DOI](https://joss.theoj.org/papers/10.21105/joss.03323/status.svg)](https://doi.org/10.21105/joss.03323)

Jenul et al., (2021). RENT: A Python Package for Repeated Elastic Net Feature Selection. Journal of Open Source Software, 6(63), 3323, https://doi.org/10.21105/joss.03323 

Bibtex entry:

    @article{RENT,
    doi = {10.21105/joss.03323},
    url = {https://doi.org/10.21105/joss.03323},
    year = {2021},
    publisher = {The Open Journal},
    volume = {6},
    number = {63},
    pages = {3323},
    author = {Anna Jenul and Stefan Schrunner and Bao Ngoc Huynh and Oliver Tomic},
    title = {RENT: A Python Package for Repeated Elastic Net Feature Selection},
    journal = {Journal of Open Source Software}
    }


