# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:59 2017

@author: Gabriel S. Gusmão, gusmaogabriels@gmail.com
"""

import os
import numpy as np
import fastMLP
import pandas as pd


os.chdir(r'C:\Users\gabris46\Desktop\Gabriel S. Gusmao\GitHub\fastMLP\Networks\Evol22[1 3 2 0 0 7 0 0 1 7]_MLP')                    
training_path = os.getcwd()+r'//Examples//MNIST\\xtraining.npz'
testing_path = os.getcwd()+r'///Examples//MNIST\\xtesting.npz'
    
#lin_classifier = fastMLP.LinearClassifier(training_path,
#                      testing_path,
#                      2/3.)  


#ann_elm = fastMLP.ELM(training_path,
#                      testing_path,
                      #2/3.) 
#ann_elm.build(3000)
#ann_elm.gen_random_layer((-0.05,0.05),np.tanh) 
#ann_elm.set_W(1e-10)
#err = ann_elm.error('test')
#print err
#pd.DataFrame(err[1]+[err[0]]).to_csv('abc')



ANN = fastMLP.MLP('Evol2'+str(np.random.randint(0,9,10)),[100,100,100,100])
ANN.load_net('Evol22[1 3 2 0 0 7 0 0 1 7]_4')
ANN.set_fold(4)
#ANN.load_data(training_path,testing_path,[])
#ANN.init_weights()
#ANN.init_folds(5)
#ANN.set_fold(0)
#ANN.train(threshold = 1.0e-5, n_itermax = 1500,rate0 = 0.25,cut = 0.25)