# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:59 2017

@author: Gabriel S. Gusmão, gusmaogabriels@gmail.com
"""

import os
import numpy as np
import fastMLP
import pandas as pd

                    
training_path = os.getcwd()+r'//Examples//MNIST\\xtraining.npz'
testing_path = os.getcwd()+r'///Examples//MNIST\\xtesting.npz'
    
#lin_classifier = fastMLP.LinearClassifier(training_path,
#                      testing_path,
#                      2/3.)  


#ann_elm = fastMLP.ELM(training_path,
#                      testing_path,
                      2/3.) 
#ann_elm.build(3000)
#ann_elm.gen_random_layer((-0.05,0.05),np.tanh) 
#ann_elm.set_W(1e-10)
#err = ann_elm.error('test')
#print err
#pd.DataFrame(err[1]+[err[0]]).to_csv('abc')



ANN = fastMLP.MLP('Evol22',[200,200,200])
ANN.load_data(training_path,testing_path,[])
ANN.init_weights()
ANN.init_folds(5)
ANN.train(threshold = 1.0e-5, n_itermax = 50,rate0 = 0.95,cut = 0.25)