# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:59 2017

@author: Gabriel S. Gusmão, gusmaogabriels@gmail.com
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

from loadMNIST import *
import matplotlib.pyplot as plt

import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "portuguese")

plt.rcParams['axes.formatter.use_locale'] = True

import seaborn as sns
import numpy as np
import os
import time
import uuid
import gc
from scipy import stats

sns.set_style('white')

for item in os.listdir(os.getcwd()):
    if item.endswith(".temp"):
        os.remove(item)

font = 'Consolas'
plt.rc('font',family=font)
plt.rc('mathtext',fontset='custom')
plt.rc('mathtext',rm=font)    
plt.rc('mathtext',it='{}:italic'.format(font))
plt.rc('mathtext',bf='{}:bold'.format(font))
plt.rc('mathtext',default='regular')
fs=8
plt.rc('font', size=fs)          # controls default text sizes
plt.rc('axes', titlesize=fs)     # fontsize of the axes title#
plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
plt.rc('legend', fontsize=fs)    # legend fontsize
plt.rc('figure', titlesize=fs)  # fontsize o

plt.rc('text.latex',unicode=True)

data = np.load('xtraining.npz')
l_train = 40000
X_tr = np.memmap('X_tr.temp',np.float32,'w+',shape=(l_train,len(data['X'][0,:])+1))
S_tr = np.memmap('S_tr.temp',np.float32,'w+',shape=(l_train,len(data['S'][0,:])))
X_tr[:,:-1] = data['X'][:l_train]
X_tr[:,-1] = 1
S_tr[:] = data['S'][:l_train]
X_v = np.memmap('X_v.temp',np.float32,'w+',shape=(data['X'].shape[0]-l_train,data['X'].shape[1]+1))
S_v = np.memmap('S_v.temp',np.float32,'w+',shape=(data['S'].shape[0]-l_train,data['S'].shape[1]))
X_v[:,:-1] = data['X'][l_train:]
X_v[:,-1] = 1
S_v[:] = data['S'][l_train:]
data.close()


data = np.load('xtesting.npz')
X_ts = np.memmap('X_ts.temp',np.float32,'w+',shape=(data['X'].shape[0],data['X'].shape[1]+1))
S_ts = np.memmap('S_ts.temp',np.float32,'w+',shape=data['S'].shape)
X_ts[:,:-1] = data['X']
X_ts[:,-1] = 1
S_ts[:] = data['S']
data.close()

class LinClassifier(object):
      
      def __init__(self,X,S,Xval,Sval,Xtest,Stest):
            self.X = X
            self.S = S
            self.Xval = Xval
            self.Sval = Sval
            self.Xtest = Xtest
            self.Stest = Stest
            self.uuid = str(uuid.uuid4())
            self.time = []
      
            self.mHtH = np.memmap('mHtH-{}.temp'.format(self.uuid),np.float32,'w+',shape=(self.X.shape[1],self.X.shape[1]))
            self.mHtH[:] = np.dot(self.X.T,self.X,self.mHtH)
            
            self.mHtS = np.memmap('mHtS-{}.temp'.format(self.uuid),np.float32,'w+',shape=(self.X.shape[1],self.S.shape[1]))
            self.mHtS[:] = np.dot(self.X.T,self.S,self.mHtS)
                    
            self.W = np.memmap('W-{}.temp'.format(self.uuid),np.float32,'w+',shape=(self.X.shape[1],self.S.shape[1]))

            self.S_dict ={}
            self.Sval_dict = {}
            self.Stest_dict = {}
            
            for i in ['','val','test']:
                  _dict = self.__getattribute__('S{}_dict'.format(i))
                  _data = self.__getattribute__('S{}'.format(i))
                  for k in range(_data.shape[1]):
                        _dict.__setitem__(k,np.where(np.argmax(_data,axis=1)==k)[0])
      
      def close(self):
            for string in ['X','S','Xval','Sval','Xtest','Stest','mHtH','mHtS','W']:
                  _ = self.__getattribute__(string)
                  fl = _.filename.rpartition('\\')[-1]
                  _._mmap.close()
                  if fl in os.listdir(os.getcwd()):
                        try:
                              os.remove(fl)
                        except:
                              pass
            
      def set_W(self,c_reg):
            t = time.time()
            self.W[:] = np.dot(np.linalg.inv(self.mHtH+c_reg*np.identity(np.shape(self.mHtH)[0])),self.mHtS)
            dt = time.time()-t
            print 'time: {}'.format(dt)
            self.time += [dt]
      
      def train(self,c_space):
            CER_val = []
            CER_val_class = []            
            for c_ in c_space:
                  print c_
                  self.set_W(c_)
                  CER, CER_class = self.error('validation')
                  CER_val.append(CER)
                  CER_val_class.append(CER_class)
            c_min = c_space[np.argmin(CER_val)]
            CER_min = np.min(CER_val)
            return c_min, CER_min, c_space, CER_val, CER_val_class
           
      def error(self,mode):
            W = self.W
            if mode == 'training':
                  X = self.X
                  Sdict = self.S_dict
            elif mode == 'validation':
                  X = self.Xval
                  Sdict = self.Sval_dict
            elif mode == 'test':
                  X = self.Xtest
                  Sdict = self.Stest_dict
            else:
                  raise Exception("mode must be one among ['training','validation','test']")
            err_class = []
            for _c in range(self.S.shape[1]):
                  err_class.append(sum((np.argmax(np.dot(X[Sdict[_c],:],W),axis=1)!=_c)*1.)/(len(Sdict[_c])))
            # sum((np.argmax(np.dot(X[np.argmax(S,axis=1)==_c],W),axis=1)!=_c)*1.0),\sum(np.argmax(S,axis=1)==_c)
            err = np.average(err_class)
            return err, err_class

class ELM(LinClassifier):
      
      def __init__(self,X,S,Xval,Sval,Xtest,Stest,n_hidden):
            LinClassifier.__init__(self,X,S,Xval,Sval,Xtest,Stest)
            self.Whidden = []
            self.n_hidden_neurons = n_hidden
            self.fun = []
            self.__status__ = False
            self.linclassifier = []
            self.make_structure(n_hidden)
                        
      def make_structure(self,n_hidden):
            self.n_hidden_neurons = n_hidden
            if not self.__status__: 
                  self.Whidden = np.memmap('Whidden-{}.temp'.format(self.uuid),np.float32,'w+',shape=(self.n_hidden_neurons,self.X.shape[1]))
                  self.linclassifier = LinClassifier(np.memmap('X-{}.temp'.format(self.uuid),np.float32,'w+',shape=(self.X.shape[0],self.n_hidden_neurons)),
                                                     self.S,
                                                     np.memmap('Xval-{}.temp'.format(self.uuid),np.float32,'w+',shape=(self.Xval.shape[0],self.n_hidden_neurons)),
                                                     self.Sval,
                                                     np.memmap('Xtest-{}.temp'.format(self.uuid),np.float32,'w+',shape=(self.Xtest.shape[0],self.n_hidden_neurons)),
                                                     self.Stest)
            else:
                  self.Whidden._mmap.resize((self.n_hidden_neurons,self.X.shape[1]))
                  self.linclassifier.X.resize((self.X.shape[0],self.n_hidden_neurons))
                  self.linclassifier.Xval.resize((self.Xval.shape[0],self.n_hidden_neurons))
                  self.linclassifier.Xtest.resize((self.Xtest.shape[0],self.n_hidden_neurons))
            
      def gen_randhiddenlayer(self,interval,fun):
            
            self.fun = fun
            self.Whidden[:] = np.random.uniform(interval[0],interval[1],(self.n_hidden_neurons,self.X.shape[1]))
            self.linclassifier.X[:] = np.dot(self.X,self.Whidden.T,self.linclassifier.X)
            self.linclassifier.X[:] = fun(self.linclassifier.X[:],self.linclassifier.X)
            self.linclassifier.Xval[:] = np.dot(self.Xval,self.Whidden.T,self.linclassifier.Xval)
            self.linclassifier.Xval[:] = fun(self.linclassifier.Xval,self.linclassifier.Xval)
            self.linclassifier.Xtest[:] = np.dot(self.Xtest,self.Whidden.T,self.linclassifier.Xtest)
            self.linclassifier.Xtest[:] = fun(self.linclassifier.Xtest,self.linclassifier.Xtest)
            self.linclassifier.mHtH[:] = np.dot(self.linclassifier.X.T,self.linclassifier.X,self.linclassifier.mHtH)
            self.linclassifier.mHtS[:] = np.dot(self.linclassifier.X.T,self.linclassifier.S,self.linclassifier.mHtS)
            self.__status__ = True
      
      def set_W(self,c_reg):
            if self.__status__:
                  self.linclassifier.set_W(c_reg)
            else:
                  print('ELM not setup')
            
      def train(self,c_space):
            if self.__status__:
                  return self.linclassifier.train(c_space)
            else:
                  print('ELM not setup')
            
      def error(self,mode):
            if self.__status__:
                  return self.linclassifier.error(mode)
            else:
                  print('ELM not setup')
            
                        

lc_ob = LinClassifier(X_tr,S_tr,X_v,S_v,X_ts,S_ts)       

i_vec = []
error_vec = []          

elm_obj = ELM(X_tr,S_tr,X_v,S_v,X_ts,S_ts,2000)
elm_obj.gen_randhiddenlayer((-0.05,0.05),np.tanh)

#out = elm_obj.train(np.logspace(-3,2,100))
#elm_obj.gen_randhiddenlayer(1500,(-0.05,0.05),np.tanh)
#out1 = elm_obj.train(np.logspace(-3,2,50))
#elm_obj.gen_randhiddenlayer(1000,(-0.05,0.05),np.tanh)
#out2 = elm_obj.train(np.logspace(-3,2,50))

      
      
      
#elm_obj.gen_randhiddenlayer(100,(-0.05,0.05),np.tanh)
#c_min, CER_min, c_space, CER_val, CER_val_class = lin_class_obj.train(c_space)
"""
plt.figure(figsize=(6,6/1.618))
m, = plt.plot(c_space,CER_val,'.',ms=3,color='black',alpha=0.2)
l, = plt.plot(c_space,CER_val,'-',lw=1,alpha=0.8)
#texts = plt.text(x=c[0],y=err_val[0]+0.0022,s="2/3",color=l.get_c()),
plt.gca().set_title(u'Erro de Validação - Holdout = 1/3')
plt.gca().set_xscale('log')
plt.gca().set_xlabel(u'$c$ em $(P^TP+cI)^{-1}$')
plt.gca().set_ylabel(u'CER\nConjunto de Validação')
plt.tight_layout()

plt.figure(figsize=(6,6/1.618))
lin_class_obj.set_W(c_min)
sns.barplot(x=range(10), y = lin_class_obj.error('test')[1],color='black')
plt.tight_layout()
"""