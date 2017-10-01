# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:59 2017

@author: Gabriel S. Gusmão, gusmaogabriels@gmail.com
"""

import os
import numpy as np
import fastMLP

                    
training_path = os.getcwd()+r'//Examples//MNIST\\xtraining.npz'
testing_path = os.getcwd()+r'///Examples//MNIST\\xtesting.npz'
    
#lin_classifier = fastMLP.LinearClassifier(training_path,
#                      testing_path,
#                      2/3.)  


#ann_elm = fastMLP.ELM(trainning_path',
#                      testing_path,
#                      2/3.) 
#ann_elm.build(1000)
#ann_elm.gen_randhiddenlayer((-0.05,0.05),np.tanh) 


ANN = fastMLP.MLP('Evol02220',[600])
ANN.load_data(training_path,testing_path,[])
ANN.init_weights()
ANN.init_folds(5)
ANN.train(threshold = 1.0e-5, n_itermax = 50,rate0 = 0.95,cut = 0.25)