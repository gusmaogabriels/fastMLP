**fastMLP**
==================================================================
*Fast batch MLP (Multi-layer Perceptron) algorithm based on numpy memmap.*

*pip install* -> `pip install --upgrade https://github.com/gusmaogabriels/fastMLP/zipball/master`

**The University of Campinas, UNICAMP**

* IA013 - Introdução à Computação Natural - *Introduction to Natural Computing*
   - *Prof. Dr. Levy Boccato; Prof. Dr. Fernando Von Zuben.*

### Overview
Multi-layer perceptron (MLP) Artificial Neural Network (ANN).  
Batch-mode training with k-fold strategy, memory-mapped files using [numpy](http://www.numpy.org/)'s [*memmap*](https://br.linkedin.com/pub/gabriel-saben%C3%A7a-gusm%C3%A3o/115/aa6/aa8).  
Training as of the [Conjugate Gradient](https://github.com/gusmaogabriels/optinpy#conjugate-gradient-methodconjugate-gradient) algorithm, exact derivatives and *H*×*p* calculations by backpropagation (BP).  

### Examples

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
		[0.041836734,
		0.024669603,
		0.212209302,
		0.128712871,
		0.102851323,
		0.265695067,
		0.083507306,
		0.142023346,
		0.224845995,
		0.197224975])
	 
#### Extreme-Learning Machine (ELM)

 ```python  
	 
# creates the ELM classifier	 
elm_classifier = mlp.ELM(training_path,testing_path,holdout=2/3.) 
# builds the a hidden layer with 200
elm_classifier.build(n_hidden=2000) 
# random uniform between -0.05 and 0.05 and hyperbolic tangent transfer function
elm_classifier.gen_random_layer(interval=(-0.05,0.05),fun=np.tanh) 
# solves the problem for the regularization parameter c = 1e-10
elm_classifier.set_W(c=1e-10)
# calculates the classification error per class and the average per class
elm_classifier.error('test') 
```
	 
	 >> (0.016326531,
		[0.010572687,
		0.055232558,
		0.044554455,
		0.039714868,
		0.043721973,
		0.025052192,
		0.059338521,
		0.05338809,
		0.073339941,
		0.042124182])

![Alt Text](/raw/elm_lin.png)	

#### Multi-layer Perceptron (MLP)

 ```python  

# instantiates the MLP class, with two hidden layers with 200 and 100 neurons, respectively	 
mlp_classifier = mlp.MLP('the_net_name',hidden_layers=[200,100]) 
# loads the data and allocates the ANN strucure memmap files under the hood
mlp_classifier.load_data(training_path,testing_path,W=[]) 
# initialize the weights, from the initial guess (if given) or uniformily random
mlp_classifier.init_weights()
# set the (k=5)-folds problem by partitioning the training set
mlp_classifier.init_folds(5) 
# begin the k-fold training session
mlp_classifier.train(threshold = 1.0e-5, n_itermax = 50,rate0 = 0.25,cut = 0.25) 
```

### Requirements
1. [numpy](http://www.numpy.org/)
2. [os](https://docs.python.org/2/library/os.html) - Miscellaneous operating system interfaces
3. [time](https://docs.python.org/2/library/time.html) - Time access and conversions
4. [tempfile](https://docs.python.org/2/library/tempfile.html) - Generate temporary files and directories
