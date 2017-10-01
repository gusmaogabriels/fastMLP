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
Training as of the [Conjugate Gradient](https://github.com/gusmaogabriels/optinpy#conjugate-gradient-methodconjugate-gradient) algorithm, exact derivatives and *H*×*p* calcuations by backpropagation (BP).  

### Examples

#### Linear Classifier

 ```python  
import fastMLP as mlp
import os
	
# paths to xtraining and xtesting npz file (derived from MNIST's idx)
# each of which includes a 'X' and 'S' arrays of data input and output, respectively
training_path = os.getcwd()+r'//Examples//MNIST\\xtraining.npz' 
testing_path = os.getcwd()+r'///Examples//MNIST\\xtesting.npz'
	 
lin_classifier = mlp.LinearClassifier(training_path,testing_path,holdout=2/3.) # creates the linear classifier
lin_classifier.build() # builds the closed-form Tikhonov regularized regression problem 
lin_classifier.set_W(c=600) # solves the problem for the regularization parameter c = 600
lin_classifier.error('test') # calculates the classification error per class and the average per class
```
	 
	 >> (0.14235765272333828,
		[0.041836734693877553,
		0.024669603524229075,
		0.21220930232558138,
		0.12871287128712872,
		0.10285132382892057,
		0.26569506726457398,
		0.083507306889352817,
		0.14202334630350194,
		0.22484599589322382,
		0.19722497522299307])
	 
#### Extreme-Learning Machine (ELM)

 ```python  
import fastMLP as mlp
import os
import numpy as np
	 
elm_classifier = mlp.ELM(training_path,testing_path,holdout=2/3.) # creates the ELM classifier
elm_classifier.build(n_hidden=2000) # builds the a hidden layer with 200
elm_classifier.gen_random_layer(interval=(-0.05,0.05),fun=np.tanh) # random uniform between -0.05 and 0.05 and hyperbolic tangent transfer function
elm_classifier.set_W(c=1e-10) # solves the problem for the regularization parameter c = 600
elm_classifier.error('test') # calculates the classification error per class and the average per class
```
	 
	 >> (0.14235765272333828,
		[0.041836734693877553,
		0.024669603524229075,
		0.21220930232558138,
		0.12871287128712872,
		0.10285132382892057,
		0.26569506726457398,
		0.083507306889352817,
		0.14202334630350194,
		0.22484599589322382,
		0.19722497522299307])

![Alt Text](/raw/elm_lin.png)	

#### Multi-layer Perceptron (MLP)

 ```python  
import fastMLP as mlp
	
	 
mlp_classifier = mlp.MLP('the_net_name',hidden_layers=[200,100]) # instantiates the MLP class, with two hidden layers with 200 and 100 neurons, respectively
mlp_classifier.load_data(training_path,testing_path,W=[]) # loads the data and allocates the ANN strucure memmap files under the hood
mlp_classifier.init_weights() # initialize the weights, from the initial guess (if given) or uniformily random
mlp_classifier.init_folds(5) # set the (k=5)-folds problem by partitioning the training set
mlp_classifier.train(threshold = 1.0e-5, n_itermax = 50,rate0 = 0.25,cut = 0.25) # begin the training session
```

### Requirements
1. [numpy](http://www.numpy.org/)
2. [os](https://docs.python.org/2/library/os.html) - Miscellaneous operating system interfaces
3. [time](https://docs.python.org/2/library/time.html) - Time access and conversions
4. [tempfile](https://docs.python.org/2/library/tempfile.html) - Generate temporary files and directories