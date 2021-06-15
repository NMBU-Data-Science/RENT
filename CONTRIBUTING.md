# Contributing to RENT

Your contribution to RENT is very welcome! For the implementantion of a new feature or bug-fixing, we encourage you to send a Pull Request to https://github.com/NMBU-Data-Science/RENT. Please add a detailed and concise
description of the invented feature or the bug. In case of fixing a bug, include comments about your solution. To improve RENT even more, feel free to send us issues with bugs, you are not sure about. 
Furthermore, you can also contribute to the improvement of the Read the Docs documentation page. We are thankful for any kind of constructive criticism and suggestions.

When you are finished with your implementantions and bug-fixing, please send a Pull Request to https://github.com/NMBU-Data-Science/RENT.

## Developing RENT
RENT can easily be developed on your computer. We recommend installing RENT in a separate environment. If you use conda, the flow could be:

1. Create and activate an environment for RENT:

```
conda create -n envRENT python=3.8
conda activate envRENT
```

2. Clone a copy of RENT from source:

```
git clone https://github.com/NMBU-Data-Science/RENT.git
```

3. Install RENT requirements:

```
cd RENT
pip install -r requirements.txt
pip install -e .
```

4. Make sure that the installation works by running the entire test suite with:

```
tox .
```

## Testing RENT
Step 4 in the previous section runs the entire test suite. Navigating to the **test** folder, you can run a test for a classification and a regression problem separately with 

```
cd test
pytest test_classification.py
pytest test_regression.py
```

## Additional comments
* RENT is written in American English
* We use Semantic Versioning (https://semver.org/)

