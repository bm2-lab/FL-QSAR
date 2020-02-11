# FL-QSAR: an horizontal federated-learning based QSAR platform
## Introduction
----
FL-QSAR is an horizontal federated-learning based QSAR framework that enables clients to simulate the scenario of _n_ clients with and without HFL. Meanwhile, FL-QSAR can be easily extended to other deep learning algorithms for solving various drug-related learning tasks.  
Our study has demonstrated the effectiveness of applying HFL in QSAR modeling. The codes of simulation for HFL in our study are in `origin_simulation_code` directory.
## Federated learning simulation
----
Dataset is randomly separated into subsets and distributed them to clients, which were then regarded as private data. Clients have the same amount of training data with chemical structure descriptors as features and bioactivities as labels and the same testing data without labels. Suppose that there are n instances in all available training data. Then a client owns a random subset of training data with 1/x · n instances, where x stands for the number of clients. As shown in the figures below, let us take the simulation for 3 clients with and without HFL as the example to illustrate the process of simulation.
__The simulation of 3 clients with and without HFL__
![](https://github.com/bm2-lab/FL-QSAR/blob/master/images/simulation.jpg)  
__The workflow of HFL__  
![](https://github.com/bm2-lab/FL-QSAR/blob/master/images/HFL.jpg)
## Install
FL-QSAR currently runs on Linux and Mac with Python >= 3.7. Windows is not supported. 
1. Clone the repository
> git clone https://github.com/bm2-lab/FL-QSAR.git  
2. Install the dependencies
> pip install torch torchvision  
> pip install crypten  
> pip install numpy  
> pip install pandas
## How FL-QSAR works
You can run the FL_QSAR.py directly as a test.
> python FL_QSAR.py

You just need to input the training data, test data and client number, and then models and performances of each single client and their collaboration via HFL will be showed in `result` directory. For illustration purpose, we took datasets __data/METAB_training.csv__ and __data/METAB_test.csv__ as an example.  
> python FL_QSAR.py -tr ./data/METAB_training.csv -te ./data/METAB_test.csv -n 4  

If you want to change more parameters, you are welcomed to change the network structure in Class `Net` and change hyperparameters in Class `Setting`. 
We set random_seed `1` to guarantee the repeatability of the result. If you want to the data sent to clients to be different each time, you can change the add `#` before `torch.manual_seed(args.seed)` or change `args.seed` to another number.
FL-QSAR can be easily extended to other deep learning algorithms for solving various drug-related learning tasks. If you want to use other algorithms such as CNN, RNN, you can change Class `Net`, Class `Setting` and function `preprocess` to meet your need.  
 Our example dataset is the METAB dataset in the Kaggle competition (Ma et al., 2015)(https://pubs.acs.org/doi/abs/10.1021/ci500747n). The original datasets can be downloaded from ci500747n_si_002.zip in the Supporting Information section. We use parameters setting of Ma et al’s article as our hyperparameters.
 ## Check the results
 You can check the results in the `result` directory after running.  
 The predicted R² for individual client as well as their collaborated one via HFL will be in result/result.txt.  Model for individual client is in model_n.pkl and model for their collaborated one via HFL is in fl_model.pkl
