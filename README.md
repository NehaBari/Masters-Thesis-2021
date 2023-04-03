# Masters-Thesis-2021

# Masters-Thesis-2021

- The aim of this project was to account for any possible changes in the gmlvq classification abilites following the incorportaion of the correction procedure (incorporated in https://github.com/rickvanveen/sklvq)


Installation: 
The installation of the standard SKLVQ package into the local repo can be found in the Masters-Thesis-2021/Installation.pdf

- The datasets used were artificial data toy dataset (to initially test the results) followed with various conditional settings for the parkinson disease dataset(not publicly accessible)

- Run the libfiles in the libfiles branch to test the functionalities for these settings 
  For example: lib_2class.py where the patients are classified as early or late parkinson diseases patients
  

- Functionalities: 
  - Processlogger : To account for conditions leading to early training
  - Model definition: The model definiton function with the necesary model parameters
  - sample, not_sampled: Functions to test the functionalities with sampled and not sampled datasets fot the minority class and to test the necessary effects
  - train_modelkfold: Repeated K-fold testing
  
