---
title: 'RENT: A Python package for repeated elastic net feature selection'
tags:
- Python
- feature selection
authors:
- name: Anna Jenul
orcid: 0000-0002-6919-3483
affiliation: 1
- name: Stefan Schrunner
  orcid: 0000-0003-1327-4855
  affiliation: 1
- name: Bao Ngoc Huynh
  orcid: 0000-0001-5210-132X
  affiliation: 2
- name: Oliver Tomic
  orcid: 0000-0003-1595-9962
  affiliation: 1
affiliations:
- name: Department of Data Science, Norwegian University of Life Sciences
  index: 1
-name: Department of Physics, Norwegian University of Life Sciences
  index: 2
date: 10.03.2021
bibliography: paper.bib

---

# Summary and Statement of Need
`RENT` is a Python package that implements the corresponding feature selection
method introduced by Jenul et al. [@Jenul:2021]. The package includes functionality
for binary classification and regression problems. RENT is based on an ensemble of
independent elastic net regularized machine learning models, trained on unique
subsets of the data. RENT provides information on
model quality, selection stability and interpretability.
Compared to established feature selection packages available in `R` and Python,
such as `Rdimtools` [@Rdimtools:2020] implementing Laplacian and Fisher scores or the
scikit-learn feature selection module [@scikit-learn] implementing recursive feature
elimination and sequential feature selection, RENT may provide a deeper understanding of the data by utilising information acquired through the ensemble. This aspect is
realized through tools for post hoc data analysis, visualization and feature selection validation provided with the package, along with an efficient and user-friendly implementation of the main methodology.

# Concept and Structure of RENT
At its core, RENT trains $K$ independent elastic net regularized models on
distinct subsets of the train dataset. The framework is demonstrated in Figure \autoref{fig:RENT}.

![Summary of RENT method [@Jenul:2021].\label{fig:RENT}](RENT_overview.png){ width=20% }

Based on three statistical criteria $\tau_1$, $\tau_2$ and $tau_3$, relevant
features are selected. While $\tau_1$ counts how often each feature was selected
over $K$ models, $\tau_2$ quantifies the stability of the feature weights --- a
feature where the $K$ weight signs alternate between positive and negative is less
stable than a feature where all weights are of constant sign. The third criterion
$\tau_3$ relies on a Student’s $t$-test whether feature weights are significantly
different from zero. The presented implementation builds on an abstract class
`RENT_Base` with a general skeleton for feature selection and post hoc analysis.
Two sub-classes RENT_Classification and RENT_Regression inherit from RENT_Base and
offer target-specific methods. The constructor of `RENT_Base` initializes the
different user-specific paramters such as the dataset, elastic net regularization
parameters or the number of models $K$. With that information, RENT starts the
training process with `train()`.

The major steps of feature selection are performed using the methods
`select_features()`, picking features with regard to cutoff criteria,
`get_summary_criteria()`, which returns a matrix with the statistical criteria
values for each feature and `get_weight_distribution()`, a matrix keeping the
exact model weights for each feature in each of the $K$ models. For initial
analysis of the results, `plot_selection_frequency()` shows a barplot of $\tau_1$,
how often each feature was selected. Additionally, two
validation studies are provided: first, a model based on random feature
selection is trained, while secondly, a model based on randomly permuted labels
of the test dataset is obtained. 
Both validation models are conducted multiple
times and results are compared to a model built with RENT features using
two Student's $t$-tests, judging the relevance of features. Empirical densities are shown in `plot_validation_study()`.


In addition to feature selection, `get_object_summary()` offers a detailed
summary about prediction accuracies for the training objects.
`plot_object_probabilites()` plots histogram for class probabilities for each training object for classification problems. Those histograms
provide additional information on prediction accuracies of training objects.
Further, for regression problems, `plot_object_errors()` relies on the same
concept with histograms about the average absolute error. `plot_object_PCA()`
performs principal component analysis and offers deeper insight into the
properties of training objects and their relation to RENT features. Principal
components and plots are generated with the `hoggorm` and `hoggormplot`
packages [@Tomic:2019].

# Ongoing Research and publication
The manuscript RENT - Repeated Elastic Net Technique for Feature Selection
is currently under review. Besides, the method and the package are used in
different master thesis projects at NMBU, mainly in the field of healthcare
data analysis.

# References
