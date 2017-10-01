# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:59 2017

@author: Gabriel S. Gusmão, gusmaogabriels@gmail.com
"""

from . import os as _os
from . import np as _np
from . import time as _time
from . import shutil as _shutil
from . import re as _re
from .import tempfile as _tf

"""
for item in _os.listdir(_os.getcwd()):
    if item.endswith(".temp"):
        _os.remove(item)

for item in _os.listdir(_os.getcwd()):
    if item.endswith(".temp"):
        _os.remove(item)
        
self._curpath = _os.getcwd()
            for _ in _os.listdir(_os.getcwd()):
                  if _re.match('_temp_*',_):
                        try:
                              _shutil.rmtree(_)
                        except:
                              print('Could not delete {}'.format(_))
                  else:
                        pass        
"""

class LinearClassifier(object):
      
      def __init__(self,train_fl,test_fl,holdout):
            
            
            self._curpath = _tf.getcwd()
            self._temp_path = _tf.mksdir(dor=self._curpath)
            _os.mkdir(self._temp_path)
            self.time = []
                       
            data = _np.load(train_fl)
            l_train = int(_np.floor(len(data['X'])*holdout))
            self.X = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(l_train,len(data['X'][0,:])+1))
            self.S = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(l_train,len(data['S'][0,:])))
            self.X[:,:-1] = data['X'][:l_train]
            self.X[:,-1] = 1
            self.S[:] = data['S'][:l_train]
            self.Xval = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(data['X'].shape[0]-l_train,data['X'].shape[1]+1))
            self.Sval= _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(data['S'].shape[0]-l_train,data['S'].shape[1]))
            self.Xval[:,:-1] = data['X'][l_train:]
            self.Xval[:,-1] = 1
            self.Sval[:] = data['S'][l_train:]
            data.close()
            
            data = _np.load(test_fl)
            self.Xtest = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(data['X'].shape[0],data['X'].shape[1]+1))
            self.Stest = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=data['S'].shape)
            self.Xtest[:,:-1] = data['X']
            self.Xtest[:,-1] = 1
            self.Stest[:] = data['S']
            data.close()
            
            self.S_dict ={}
            self.Sval_dict = {}
            self.Stest_dict = {}
            
            for i in ['','val','test']:
                  _dict = self.__getattribute__('S{}_dict'.format(i))
                  _data = self.__getattribute__('S{}'.format(i))
                  for k in range(_data.shape[1]):
                        _dict.__setitem__(k,_np.where(_np.argmax(_data,axis=1)==k)[0])
            
            self.mHtH = []
            self.mHtS = []
            self.W = []
     
      def build(self):
            self.mHtH = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.X.shape[1],self.X.shape[1]))
            self.mHtH[:] = _np.dot(self.X.T,self.X,self.mHtH)
            
            self.mHtS = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.X.shape[1],self.S.shape[1]))
            self.mHtS[:] = _np.dot(self.X.T,self.S,self.mHtS)
                    
            self.W = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.S.shape[1],self.X.shape[1]))

      
      def close(self):
            for string in ['X','S','Xval','Sval','Xtest','Stest','mHtH','mHtS','W']:
                  _ = self.__getattribute__(string)
                  fl = _.filename.rpartition('\\')[-1]
                  _._mmap.close()
                  if fl in _os.listdir(_os.getcwd()):
                        try:
                              _os.remove(fl)
                        except:
                              pass
      def predict(self,x):
            y = _np.dot(x,self.W[:-1,:])+self.W[-1,:]
            return _np.argmax(y,axis=1), y
      
      def set_W(self,c_reg):
            t = _time.time()
            self.W[:] = _np.dot(_np.linalg.inv(self.mHtH+c_reg*_np.identity(_np.shape(self.mHtH)[0])),self.mHtS).T
            dt = _time.time()-t
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
            c_min = c_space[_np.argmin(CER_val)]
            CER_min = _np.min(CER_val)
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
                  err_class.append(sum((_np.argmax(_np.dot(X[Sdict[_c],:],W.T),axis=1)!=_c)*1.)/(len(Sdict[_c])))
            # sum((_np.argmax(_np.dot(X[_np.argmax(S,axis=1)==_c],W),axis=1)!=_c)*1.0),\sum(_np.argmax(S,axis=1)==_c)
            err = _np.average(err_class)
            return err, err_class

class ELM(LinearClassifier):
      
      def __init__(self,train_fl,test_fl,holdout):
            self._curpath = _tf.getcwd()
            self._temp_path = _tf.mksdir(dor=self._curpath)
            self.time = []
            self.n_hidden_neurons = []
            self.fun = []
            self.__status__ = False
            self.linclassifier = LinearClassifier(train_fl,test_fl,holdout)
            self.W = []
            self.S_dict = self.linclassifier.S_dict
            self.Sval_dict = self.linclassifier.Sval_dict
            self.Stest_dict = self.linclassifier.Stest_dict
            self.Whidden = []
            self.X = []
            self.Xval = []
            self.Xtest = []
            self.mHtH = []
            self.mHtS = []
            self.S = self.linclassifier.S
            self.Sval = self.linclassifier.Sval
            self.Stest = self.linclassifier.Stest
                        
      def build(self,n_hidden):
            self.n_hidden_neurons = n_hidden
            _ = self.linclassifier
            if not self.__status__: 
                  self.Whidden = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden_neurons,_.X.shape[1]))
                  self.W = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.S.shape[1],self.n_hidden_neurons))
                  self.X = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(_.X.shape[0],self.n_hidden_neurons))
                  self.Xval = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(_.Xval.shape[0],self.n_hidden_neurons))
                  self.Xtest = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(_.Xtest.shape[0],self.n_hidden_neurons))
                  self.mHtH = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden_neurons,self.n_hidden_neurons))
                  self.mHtS = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden_neurons,_.S.shape[1]))
            else:
                  self.Whidden._mmap.resize((self.n_hidden_neurons,_.X.shape[1]))
                  self.W._mmap.resize((self.S.shape[1],self.n_hidden_neurons))
                  self.X.resize((_.X.shape[0],self.n_hidden_neurons))
                  self.Xval.resize((_.Xval.shape[0],self.n_hidden_neurons))
                  self.Xtest.resize((_.Xtest.shape[0],self.n_hidden_neurons))
                  self.mHtH.resize((self.n_hidden_neurons,self.n_hidden_neurons))
                  self.mHtS.resize((self.n_hidden_neurons,_.S.shape[1]))
            
      def gen_randhiddenlayer(self,interval,fun):
            
            self.fun = fun
            self.Whidden[:] = _np.random.uniform(interval[0],interval[1],(self.n_hidden_neurons,self.linclassifier.X.shape[1]))
            self.X[:] = _np.dot(self.linclassifier.X,self.Whidden.T,self.X)
            self.X[:] = fun(self.X[:],self.X)
            self.Xval[:] = _np.dot(self.linclassifier.Xval,self.Whidden.T,self.Xval)
            self.Xval[:] = fun(self.Xval,self.Xval)
            self.Xtest[:] = _np.dot(self.linclassifier.Xtest,self.Whidden.T,self.Xtest)
            self.Xtest[:] = fun(self.Xtest,self.Xtest)
            self.mHtH[:] = _np.dot(self.X.T,self.X,self.mHtH)
            self.mHtS[:] = _np.dot(self.X.T,self.S,self.mHtS)
            self.__status__ = True
      
      
      def set_W(self,c_reg):
            if self.__status__:
                  t = _time.time()
                  self.W[:] = _np.dot(_np.linalg.inv(self.mHtH+c_reg*_np.identity(_np.shape(self.mHtH)[0])),self.mHtS).T
                  dt = _time.time()-t
                  print 'time: {}'.format(dt)
                  self.time += [dt]
            else:
                  print('ELM not setup')
            
      def train(self,c_space):
            if self.__status__:
                  return super(ELM,self).train(c_space)
            else:
                  print('ELM not setup')
            
      def error(self,mode):
            if self.__status__:
                  return super(ELM,self).error(mode)
            else:
                  print('ELM not setup')
                  
      def predict(self,x):
            if any([i == self.linclassifier.X.shape[1]-1 for i in x.shape]):
                  x = _np.concatenate((x,_np.ones([x.shape[0],1])),axis=1)
            else:
                  raise Exception('Input variable must be of size {}'.format(self.linclassifier.X.shape[1]-1))
            y = _np.dot(self.fun(_np.dot(x,self.Whidden.T)),self.W)
            return _np.argmax(y,axis=1), y
            
     
"""
i_vec = []
error_vec = []          

elm_obj = ELM(X_tr,S_tr,X_v,S_v,X_ts,S_ts,2000)
elm_obj.gen_randhiddenlayer((-0.05,0.05),_np.tanh)

#out = elm_obj.train(_np.logspace(-3,2,100))
#elm_obj.gen_randhiddenlayer(1500,(-0.05,0.05),_np.tanh)
#out1 = elm_obj.train(_np.logspace(-3,2,50))
#elm_obj.gen_randhiddenlayer(1000,(-0.05,0.05),_np.tanh)
#out2 = elm_obj.train(_np.logspace(-3,2,50))

      
      
      
#elm_obj.gen_randhiddenlayer(100,(-0.05,0.05),_np.tanh)
#c_min, CER_min, c_space, CER_val, CER_val_class = lin_class_obj.train(c_space)

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