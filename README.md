# A synthetic data generator for particle physics using variational autoencoders
  In this project, we augmented Monte-Carlo simulated data for charged Higgs boson searches. Events are selected if they have two light leptons (electron or muon) of the same sign and exactly one hadronically decaying tau-lepton. For the data generation, a variational autoencoder model was used with evidence lower bound and symmetric equilibrium learning. Both mentioned learning approaches were also tested with hierarchical (Ladder) architecture. For the data quality assessment, both qualitative and quantitative metrics were taken into account. The standard evidence lower bound (ELBO) learning model was selected as the best-performing option. The model was then used to generate data for the signal and background separation analysis experiments. The dependence of classifier performance on the training dataset size was demonstrated using two widely used machine learning paradigms for tabular data classification: gradient-boosted decision trees and deep neural networks.


## Installation
### TODO


## Usage
### On lxplus server
  To perform the analysis, data needs to be properly preprocessed. It means to apply preselection and compute weights for every event. Moreover, data partitioned by year of data-taking needs to be conflated and partitioned again into physical reaction (tbH+, ttH, ttW, ttZ, tt, VV, other) datasets.     

The preprocessing part has to be done on the lxplus server since the root ntuples are too large to be copied. The command
```python
python preprocess_rdf.py
```
runs the script that produces preprocessed data for each reaction. The datasets can be used for training the generator and classifier. The signal class (tbH+) includes 6 masses. To produce a dataset with combined signal masses one can use
```python
python dataset_combine.py
```
command.

## On private machine
To prepare two datasets for binary classification where all the backgrounds are conflated into one the 
```python
python classification_preprocess.py
```
can be run.

### Data augmentation
One of the approaches listed below can be chosen to generate new artificial samples.
Implemented generative models:
  -  Standard VAE with ELBO learning
  -  Ladder VAE with ELBO learning
  -  Standard VAE with Symmetric Equilibrium Learning
  -  Ladder VAE with Symmetric Equilibrium Learning
  -  Deep Diffusion Generative Model (Still in progress) 

The command
```python
python generate.py
```
can be used to generate an arbitrary number of new samples. Separated sampling can be performed if the chosen model is already trained. 
### Data classification
Implemented classifiers:
 - MLP
 - XGboost

