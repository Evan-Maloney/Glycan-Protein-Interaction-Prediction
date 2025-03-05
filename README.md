# Glycan-Protein-Interaction-Prediction


# Installing miniconda 

MiniConda is a minimal anaconda env that just has python and conda installed. We will use this to create a specific env for this project and only install the libraries we need. 

Download of miniconda: https://www.anaconda.com/download/
(Scroll down to very bottom where it says MiniConda **Not** Anaconda)

When installed you should be able to run conda in your terminal.

Outputs a list of your current enviornments 
```
conda env list
``` 

Create a new enviornment for this project
```
conda create --name glycan-env
```

To switch to this enviornment from your base enviornment 
```
source activate glycan-env
``` 


Then install the following packages in this conda env:
```
conda install pandas

conda install -c conda-forge rdkit

conda install ipykernel

conda install conda-forge::biopython
```

Run this to see all the packages installed in your current enviornment. Look for pandas, rdkit, ipykernel. (The rest of the packages are dependencies of these packages that were installed as well)
```
conda list
```

# Selecting the glycan-env in jupyter notebooks

When creating a notebook, you must click on the following:

1. Select Kernel
2. Select Another Kernel (Optional as it might recomend to select your default local/bin python env)
3. Python Enviornments
4. glycan-env

This will ensure the jupyter notebook is using your glycan env to check for python packages when you import them