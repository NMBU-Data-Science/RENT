---
title: 'RENT: A Python Package for Repeated Elastic Net Feature Selection'
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
 - name: Department of Physics, Norwegian University of Life Sciences
   index: 2
date: 10.03.2021
bibliography: paper.bib

---

# Summary and Statement of Need
RENT is a Python package that implements the feature selection method of the same name [@Jenul:2021]. The package includes functionality for binary classification and regression problems. RENT is based on an ensemble of elastic net regularized models, which are trained on unique subsets of the data. Along with selecting informative features, the method provides information on model quality, selection stability, as well as interpretability. Compared to established feature selection packages available in `R` and Python, such as `Rdimtools` [@Rdimtools:2020] implementing Laplacian and Fisher scores or the scikit-learn feature selection module [@scikit-learn] implementing recursive feature elimination and sequential feature selection, RENT creates a deeper understanding of the data by utilising information acquired through the ensemble. This aspect is realized through tools for post hoc data analysis, visualization and feature selection validation provided with the package, along with an efficient and user-friendly implementation of the main methodology.

# Concept and Structure of RENT
At its core, RENT trains $K$ independent elastic net regularized models on distinct subsets of the train dataset. The framework is demonstrated in \autoref{fig:RENT}.

![Summary of RENT method [@Jenul:2021].\label{fig:RENT}](images/RENT_overview.png)

Based on three statistical cutoff criteria $\tau_1$, $\tau_2$ and $\tau_3$, relevant features are selected. While $\tau_1$ counts how often each feature was selected over $K$ models, $\tau_2$ quantifies the stability of the feature weights --- a feature where the $K$ weight signs alternate between positive and negative is less stable than a feature where all weights are of constant sign. The third criterion $\tau_3$ deploys a Student’s $t$-test to judge whether feature weights are significantly different from zero. The presented implementation builds on an abstract class `RENT_Base` with a general skeleton for feature selection and post hoc analysis. Two inherited classes, `RENT_Classification` and `RENT_Regression`, offer target-specific methods. The constructor of `RENT_Base` initializes the different user-specific paramters such as the dataset, elastic net regularization parameters or the number of models $K$.
After training, feature selection is conducted by use of the cutoff criteria. Deeper insights are provided by a matrix containing the cutoff criteria values of each feature, as well as a matrix comprising raw model weights of each feature throughout the $K$ elementary model. For initial analysis of the results, the package delivers multiple plotting functions, such as a barplot of $\tau_1$. Additionally, two validation studies are implemented: first, a model based on random feature selection is trained, while second, a model based on randomly permuted labels of the test dataset is obtained. Results of both validation models are compared to a model built with RENT features using Student's $t$-tests as well as empirical densities.

In addition to feature selection, RENT offers a detailed summary about prediction accuracies for the training objects. For each training object, this information can be visualized as histograms of class probabilities for classification problems or histograms of mean absolute errors for regression problems, respectively. For extended analysis,  principal component analysis reveals properties of training objects and their relation to features selected by RENT. For computation and visualization of principal components, RENT uses functionality from the `hoggorm` and `hoggormplot` packages [@Tomic:2019].

# Ongoing Research and Dissemination
The manuscript RENT - Repeated Elastic Net Technique for Feature Selection is currently under review. Further, the method and the package are used in
different master thesis projects at the Norwegian University of Life Sciences, mainly in the field of healthcare data analysis.

# Acknowledgements
We thank Runar Helin for proofreading of the documentation.

# References
