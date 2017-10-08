**fastMLP**
==================================================================
*Fast batch MLP (Multi-layer Perceptron) algorithm using numpy memmap.*

*pip install* -> `pip install --upgrade https://github.com/gusmaogabriels/fastMLP/zipball/master`

**The University of Campinas, UNICAMP**

* IA013 - Introdução à Computação Natural - *Introduction to Natural Computing*
   - *Prof. Dr. Levy Boccato; Prof. Dr. Fernando Von Zuben.*

### Overview
Fully Connected Multi-layer perceptron (MLP) Artificial Neural Network (ANN).  
Batch-mode training with k-fold strategy, memory-mapped files using [numpy](http://www.numpy.org/)'s [*memmap*](https://br.linkedin.com/pub/gabriel-saben%C3%A7a-gusm%C3%A3o/115/aa6/aa8).  
Training as of the [Conjugate Gradient](https://github.com/gusmaogabriels/optinpy#conjugate-gradient-methodconjugate-gradient) algorithm, exact derivatives and *H*×*p* calculations by backpropagation (BP).  

### Examples

#### **A fully connected two-hidden layer MLP ANN with 4 and 3 neuron in each, 3 inputs and 3 outputs.**

 ```python  
import fastMLP as mlp

# instantiates the MLP class, with two hidden layers with 4 and 3 neurons in each	 
mlp_classifier = mlp.MLP('myMLP',hidden_layers=[4,3]) 
```

![Alt Text](/raw/MLP.png)

#### **MNIST Handwritten Digits Classification**
*from MNIST [sample digits database](/Examples/MNIST)*

#### Linear Classifier

 ```python  
import fastMLP as mlp
import os
	
# paths to xtraining and xtesting npz file (derived from MNIST's idx)
# each of which includes a 'X' and 'S' arrays of data input and output, respectively
training_path = os.getcwd()+r'//Examples//MNIST\\xtraining.npz' 
testing_path = os.getcwd()+r'///Examples//MNIST\\xtesting.npz'

# creates the linear classifier
lin_classifier = mlp.LinearClassifier(training_path,testing_path,holdout=2/3.)
# builds the closed-form Tikhonov regularized regression problem 
lin_classifier.build()
# solves the problem for the regularization parameter c = 600
lin_classifier.set_W(c=600) 
# calculates the classification error per class and the average per class
lin_classifier.error('test') 
```
	 
	 >> (0.142357652,
		[0.0418367,0.0246696,0.2122093,0.1287128,0.1028513,0.2656950,0.0835073,0.1420233,0.2248459,0.1972249])
	 
#### Extreme-Learning Machine (ELM)

 ```python  
	 
# creates the ELM classifier	 
elm_classifier = mlp.ELM(training_path,testing_path,holdout=2/3.) 
# builds the a hidden layer with 2000 neurons
elm_classifier.build(n_hidden=2000) 
# random uniform between -0.05 and 0.05 and hyperbolic tangent transfer function
elm_classifier.gen_random_layer(interval=(-0.05,0.05),fun=np.tanh) 
# solves the problem for the regularization parameter c = 1e-10
elm_classifier.set_W(c=1e-10)
# calculates the classification error per class and the average per class
elm_classifier.error('test') 
```
	 
	 >> (0.0163265,
		[0.0105726,0.0552325,0.0445544,0.0397148,0.0437219,0.0250521,0.0593385,0.053388,0.0733399,0.0421241])

![Alt Text](/raw/elm_lin.png)	

#### Multi-layer Perceptron (MLP)

 ```python  

# instantiates the MLP class as a classifier
mlp_classifier = mlp.MLP('the_net_name', mode='classification')
# with two hidden layers with 200 and 100 neurons, respectively	 
mlp_classifier.set_structure([200,100])
# loads the data and allocates the ANN strucure memmap files under the hood
mlp_classifier.load_data(training_path,testing_path,W=[]) 
# initialize the weights, from the initial guess (if given) or uniformily random
mlp_classifier.init_weights()
# set the (k=5)-folds problem by partitioning the training set
mlp_classifier.init_folds(5) 
# begin the k-fold training session
mlp_classifier.train(threshold = 1.0e-5, n_itermax = 1000, rate0 = 0.25,cut = 0.25) 
```

![Alt Text](/raw/MLP2.png)	

### Requirements
1. [numpy](http://www.numpy.org/)
2. [os](https://docs.python.org/2/library/os.html) - Miscellaneous operating system interfaces
3. [time](https://docs.python.org/2/library/time.html) - Time access and conversions
4. [tempfile](https://docs.python.org/2/library/tempfile.html) - Generate temporary files and directories

Copyright © 2016 - Gabriel Sabença Gusmão

[![linkedin](https://static.licdn.com/scds/common/u/img/webpromo/btn_viewmy_160x25.png)](https://br.linkedin.com/pub/gabriel-saben%C3%A7a-gusm%C3%A3o/115/aa6/aa8)

[![researchgate](https://www.researchgate.net/images/public/profile_share_badge.png)](https://www.researchgate.net/profile/Gabriel_Gusmao?cp=shp)
