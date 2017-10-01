# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:59 2017

@author: Gabriel S. Gusmão, gusmaogabriels@gmail.com
"""

from . import os as _os
from . import shutil as _shutil
from . import np as _np
from . import re as _re
from . import tempfile as _tf
from . import time as _time

        
class MLP(object):
    
    class _wrapper(object):
        
        def __init__(self,buffer,var,offset,delta,mid=True):
            self.__buffer__ = buffer
            self.var = var
            self.offset = offset
            self.delta = delta
            self.mid = mid
            
        def __getitem__(self,_):
            if self.delta > 0:
                if self.mid:
                    return self.__buffer__[self.var][_+self.offset][:,:-self.delta]
                else:
                    return self.__buffer__[self.var][:,:-self.delta]
            else:
                if self.mid:
                    return self.__buffer__[self.var][_+self.offset][:,:]
                else:
                    return self.__buffer__[self.var][:,:]
        
    def __init__(self,
            filename,
            n_hidden = [5,2]):
        
        self._curpath = _os.getcwd()
        #self._temp_path = _tf.mkdtemp(dir=self._curpath)
        """
        for _ in _os.listdir(_os.getcwd()):
              if _re.match('_temp_*',_):
                  try:
                        _shutil.rmtree(_)
                  except:
                        print('Could not delete {}'.format(_))
              else:
                    pass
        """
        if 'Networks' in _os.listdir(_os.getcwd()):
              if '{}_MLP'.format(filename) in _os.listdir(_os.getcwd()+'\\Networks'):
                    raise Exception('Filename already exists')
              else:
                    _os.mkdir('Networks/{}_MLP'.format(filename)) 
        else:
              _os.mkdir('Networks'.format(filename))
              _os.mkdir('Networks/{}_MLP'.format(filename))
        #_os.mkdir(self._temp_path+'/XYtrain')
        #_os.mkdir(self._temp_path+'/XYtest')
        #_os.mkdir(self._temp_path+'/XY')
        #_os.mkdir(self._temp_path+'/W')
          
        # Loading the training data (it is assumed that the range of the
        # training data is adequate. If not, they have to be normalized).
        # X (input matrix [N]x[m]) and S (output matrix [N]x[n_out])
        self.data_training = []
        self.data_testing = []
        self.data_weights = []
        self.__buffer__ = {}
        self.__shapes__ = {}        
        self.Ntotal = 0
        self.n_in = 0
        self.n_out = 0
        self.Ntet = 0
        
        # Set References
        self.Xtraining = []
        self.Xvalidation = []
        self.Xtesting = []
        self.Straining = []
        self.Svalidation = []
        self.Stesting = []
        
        # Set sizes
        self.Ntraining = 0
        self.Nvalidation = 0
        
        # ANN parametric values
        self.n_hidden = n_hidden
        self.out_layer = len(self.n_hidden)
        
        # Optimization aprameters
        self.k_folds = 0
        
        self.__name__ = filename
        # k-Fold results dictionary        
        """
        old names
        """
        self.fold_dict_generator = lambda : {'CERv':[],
                                        'CER_min':[],
                                        'stw':[],
                                        'rms_w':[],                                        
                                        'epoch':0, # epoch
                                        'niter_v':0,
                                        'error_per_class_v':[],
                                        'eq':[],
                                        'training_time':[],
                                        'indices':[]}
        ### kfold
        self.randperm = []    # random permutation for the training set
        self.__folders__ = {}
        self.__current_fold__ = [] # current folder
        
        self.__isloaded__ = False
   
        ### Create buffer
        self.__buffer__.__setitem__('Wopt',{})        # Best W
        self.__buffer__.__setitem__('Winit',{})       # Iinitial W
        self.__buffer__.__setitem__('W',{})           # Current W
        self.__buffer__.__setitem__('W0',{})          # Last iteration W
        self.__buffer__.__setitem__('dW',{})          # Current dW
        self.__buffer__.__setitem__('dW0',{})         # Last iteration dW
        self.__buffer__.__setitem__('rW',{})          # W residue
        self.__buffer__.__setitem__('error',{})       # Layer errors 
        self.__buffer__.__setitem__('rerror',{})      # Layer error2
        self.__buffer__.__setitem__('XY',{})          # X/Y allocation for the training set
        self.__buffer__.__setitem__('rXY',{})         # X/Y allocation (R) for the training set
        self.__buffer__.__setitem__('XYval',{})       # X/Y allocation for the validation set
        self.__buffer__.__setitem__('XYtest',{})      # X/Y allocation for the testing set       
        
        # Buffer wrapper creation for intra-class opeartions by reference
        self.X = MLP._wrapper(self.__buffer__,'XY',0,0)
        self.Y = MLP._wrapper(self.__buffer__,'XY',1,1)
        self.rX = MLP._wrapper(self.__buffer__,'rXY',0,0)
        self.rY = MLP._wrapper(self.__buffer__,'rXY',1,1)
        self.Xval = MLP._wrapper(self.__buffer__,'XYval',0,0)
        self.Yval = MLP._wrapper(self.__buffer__,'XYval',1,1)
        self.Xtest = MLP._wrapper(self.__buffer__,'XYtest',0,0)
        self.Ytest = MLP._wrapper(self.__buffer__,'XYtest',1,1)
        self.Wopt = MLP._wrapper(self.__buffer__,'Wopt',0,0)
        self.W = MLP._wrapper(self.__buffer__,'W',0,0)
        self.W0 = MLP._wrapper(self.__buffer__,'W0',0,0)
        self.dW = MLP._wrapper(self.__buffer__,'dW',0,0)
        self.dW0 = MLP._wrapper(self.__buffer__,'dW0',0,0)
        self.rW = MLP._wrapper(self.__buffer__,'rW',0,0)
        self.error = MLP._wrapper(self.__buffer__,'error',0,0)
        self.rerror = MLP._wrapper(self.__buffer__,'rerror',0,0)
        self.S = MLP._wrapper(self.__buffer__,'S',0,0,mid=False)
        
        # for fast definition of solving scope within functions
        self._partition_wrapper = {'validation':[],
                                   'training':[],
                                   'testing': []}
        
        # Calculation flags (so as not to evaluate the network unnecessarily)
        self.__haschanged__ = {'validation':True,'training':True,'testing':True}
        self.__class_position__ = {'validation':{},'training':{},'testing':{}}
        
        self.transf_fun = _np.tanh
        self.transf_fund = lambda y : 1.-y**2

    def load_data(self,data_training,data_testing,data_weights=[]):
        self.data_training = data_training
        _ = _np.load(data_training,mmap_mode='r')
        self.Ntotal = _np.shape(_['X'])[0]
        self.n_in = _np.shape(_['X'])[1]
        self.n_out = _np.shape(_['S'])[1]
        for i in ['X','S']:
            self.__buffer__.__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=_np.shape(_[i])))
            self.__buffer__[i][:] = _[i]
        _.close()
        self.data_testing = data_testing
        _ = _np.load(data_testing,mmap_mode='r')
        self.Ntest  = _np.shape(_['X'])[0]
        for i in ['X','S']:
            self.__buffer__.__setitem__(i+'test',_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=_np.shape(_[i])))
            self.__buffer__[i+'test'][:] = _[i]
        _.close()
        self.Xtesting = self.__buffer__['Xtest']
        self.Stesting = self.__buffer__['Stest']
        list0 = _np.argmax(self.Stesting,axis=1)
        for i in range(len(self.Stesting[0,:])):
              self.__class_position__['testing'].__setitem__(i,set(_np.where(list0==i)[0]))
        self.__buffer__['XYtest'].__setitem__(0,self.Xtesting)
        self._partition_wrapper['testing'] = [self.Xtest,self.Ytest,self.Stesting]
        
        ### allocate W, dW and rW spaces
        n_in = self.n_in
        for i in range(len(self.n_hidden)):
              self.__buffer__['Wopt'].__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden[i],n_in+1)))
              self.__buffer__['Winit'].__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden[i],n_in+1)))
              self.__buffer__['W'].__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden[i],n_in+1)))
              self.__buffer__['W0'].__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden[i],n_in+1)))
              self.__buffer__['dW'].__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden[i],n_in+1)))
              self.__buffer__['dW0'].__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden[i],n_in+1)))
              self.__buffer__['rW'].__setitem__(i,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_hidden[i],n_in+1)))
              n_in = self.n_hidden[i]
        self.__buffer__['Wopt'].__setitem__(i+1,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_out,self.n_hidden[-1]+1)))
        self.__buffer__['Winit'].__setitem__(i+1,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_out,self.n_hidden[-1]+1)))
        self.__buffer__['W'].__setitem__(i+1,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_out,self.n_hidden[-1]+1)))
        self.__buffer__['W0'].__setitem__(i+1,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_out,self.n_hidden[-1]+1)))
        self.__buffer__['dW'].__setitem__(i+1,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_out,self.n_hidden[-1]+1)))
        self.__buffer__['dW0'].__setitem__(i+1,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_out,self.n_hidden[-1]+1)))
        self.__buffer__['rW'].__setitem__(i+1,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.n_out,self.n_hidden[-1]+1)))
        self.stw = dict([[i,[]] for i in range(len(self.n_hidden)+1)])
        if data_weights:
              self.data_weights = data_weights
        else:
              pass
        
    def init_folds(self,n_folds):
        self.k_folds = n_folds
        self.__folders__ = dict([[i,self.fold_dict_generator()] for i in range(self.k_folds)])   # k-folds results dictionary (per fold)
        self.randperm = _np.random.permutation(self.Ntotal) # random partitionsing
        n = self.Ntotal/self.k_folds
        for i in range(self.k_folds-1):
            self.__folders__[i]['indices'] = self.randperm[n*i:n*(i+1)]
        self.__folders__[i+1]['indices'] = self.randperm[n*(self.k_folds-1):]
            
    def set_fold(self,folder):
        for i in [self.__getattribute__(j) for j in ['Xtraining','Straining','Xvalidation','Svalidation']]:
              if len(i):
                    i._mmap.close()
              else:
                    pass
        for j in ['XY','rXY','XYval','XYtest','error','rerror']:
              for i in self.__buffer__[j].values():
                    if len(i):
                          i._mmap.close()
                    else:
                          pass
        self.__current_fold__ = folder
        ind_training = set()
        for i in set(range(self.k_folds))^{self.__current_fold__}:
            ind_training ^= set(self.__folders__[i]['indices'])
        ind_validation = set(self.__folders__[self.__current_fold__]['indices'])
        self.Ntraining = len(ind_training)
        self.Nvalidation = len(ind_validation)
        
        # memory mapping for the  training and validation sets
        self.Xtraining = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntraining,self.n_in+1))
        self.Xtraining[:,:-1] = self.__buffer__['X'][list(ind_training),:]
        self.Xtraining[:,-1] = 1 
        self.Straining = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntraining,self.n_out))
        self.Straining[:] = self.__buffer__['S'][list(ind_training),:]
        self.Xvalidation = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Nvalidation,self.n_in+1))
        self.Xvalidation[:,:-1] = self.__buffer__['X'][list(ind_validation),:]
        self.Xvalidation[:,-1] = 1
        self.Svalidation = _np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Nvalidation,self.n_out))
        self.Svalidation[:] = self.__buffer__['S'][list(ind_validation),:]
        
        # assign mapped references their respect buffer dictionaries
        self.__buffer__['XY'].__setitem__(0,self.Xtraining)
        self.__buffer__['XYval'].__setitem__(0,self.Xvalidation)
        self.__buffer__['rXY'].__setitem__(0,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntraining,self.n_in+1)))
        ##############self.__buffer__['rXY'][0][:,-1] = 1
        self._partition_wrapper['validation'] =  [self.Xval,self.Yval,self.Svalidation]
        self._partition_wrapper['training'] =  [self.X,self.Y,self.Straining]
        for k in [[self.Svalidation,'validation'],[self.Straining,'training']]:
              list0 = _np.argmax(k[0],axis=1)
              for i in range(len(k[0][0,:])):
                    self.__class_position__[k[1]].__setitem__(i,set(_np.where(list0==i)[0].tolist()))
        j = 1
        for n in self.n_hidden+[self.n_out]:
            self.__buffer__['XY'].__setitem__(j,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntraining,n+1)))
            self.__buffer__['rXY'].__setitem__(j,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntraining,n+1)))
            self.__buffer__['XYval'].__setitem__(j,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Nvalidation,n+1)))
            self.__buffer__['XYtest'].__setitem__(j,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntest,n+1)))
            self.__buffer__['error'].__setitem__(j,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntraining,n)))
            self.__buffer__['rerror'].__setitem__(j,_np.memmap(_tf.NamedTemporaryFile(),_np.float32,'w+',shape=(self.Ntraining,n)))
            self.__buffer__['XY'][j][:,-1] = 1
            self.__buffer__['rXY'][j][:,-1] = 1
            self.__buffer__['XYval'][j][:,-1] = 1
            self.__buffer__['XYtest'][j][:,-1] = 1
            j+=1
            

    def init_weights(self):
        if not self.data_weights:
            for k in self.__buffer__['W'].keys():
                self.__buffer__['W'][k][:] = -0.1 + 0.2*_np.random.rand(*self.__buffer__['W'][k].shape)
                self.__buffer__['W0'][k][:] = self.__buffer__['W'][k][:]
                self.__buffer__['Winit'][k][:] = self.__buffer__['W'][k][:]
        else:
            _ = _np.load(self.data_weights,mmap_mode='r')['W']
            for i in _.keys():
                  self.__buffer__['W'][i] = _[i]
                  self.__buffer__['W0'][i] = _[i]
                  self.__buffer__['Winit'][i] = _[i]
            _.close()
                  
    def reset_weights(self):  
        for k in self.__buffer__['W'].keys():
              self.__buffer__['W'][k][:] = self.__buffer__['Winit'][k][:]
              self.__buffer__['W0'][k][:] = self.__buffer__['Winit'][k][:]
                     
    def f_CER(self,mode):
        """"
        mode = 'training' or 'validation' or 'test'
        """
        if mode in ('training','validation','testing'):
            if self.__haschanged__[mode]:
                self.eval_net(mode)
            else:
                pass
            x, y, S = self._partition_wrapper[mode]
            CER_c = {}
            for i in self.__class_position__[mode].keys():
                  ind = list(self.__class_position__[mode][i])
                  CER_c.__setitem__(i,1.-sum((_np.argmax(y[self.out_layer][ind],axis=1)==_np.argmax(S[ind],axis=1))*1.)/float(len(ind)))
            CER = _np.mean(CER_c.values())
            return CER, CER_c           
        else:
            raise Exception("'mode = 'training' or 'validation' or 'testing'")
            
    def save_net(self,filename):
          _np.savez_compressed(filename,
                              name = __name__,
                              data={'W':self.__buffer__['W'],
                                    'Wopt':self.__buffer__['Wopt'],
                                    'Winit':self.__buffer__['Winit'],
                                    'XStesting':self.data_testing,
                                    'XStraining':self.data_training},
                              kfolds= self.__folders__,
                              init = self.n_hidden,
                              folds = {'n_folds':self.k_folds,
                                     'randperm':self.randperm})
          
    def save_weights(self,filename):
          _np.savez_compressed(filename,Wopt=self.__buffer__['Wopt'])

    def load_net(self,filename):
         data = _np.load(filename+'.npz')
         #self.__name__ = data['name']
         self.__init__(**data['init'].tolist())
         self.load_data(*[data['data'].tolist()['XStraining'],data['data'].tolist()['XStesting'],data['data'].tolist()['Winit']])
         for i in data['data'].tolist()['Wopt'].keys():
               self.__buffer__['Wopt'][i][:] = data['data'].tolist()['Wopt'][i]
         for i in data['data'].tolist()['W'].keys():
               self.__buffer__['W'][i][:] = data['data'].tolist()['W'][i] 
         self.___folds__= data['kfolds'].tolist().tolist()
         for i in data['folds'].tolist().keys():
               self.__setattr__(i,data['folds'].tolist()[i])
         self.__isloaded__ = True
         
    def reset_status(self):
          if self.__isloaded__:
                data = _np.load(self.__name__+'.npz')
                self.__folders__ = data['kfolds'].tolist()
          else:
                self.__folders__ = self.__folders__ = dict([[i,self.fold_dict_generator()] for i in range(self.k_folds)])   # k-folds results dictionary (per fold)
       
    def eval_net(self, mode):
        """"
        mode = 'training' or 'validation' or 'testing'
        """
        w = self.W
        if mode in ('training','validation','testing'):
            if self.__haschanged__[mode]:        
                x, y, S = self._partition_wrapper[mode]
                for i in range(self.out_layer):
                    y[i][:] = self.transf_fun(x[i].dot(w[i].T),y[i][:]) # hidden layers
                y[self.out_layer][:] = x[self.out_layer].dot(w[self.out_layer].T) # output layer
                self.__haschanged__[mode] = False      
        else:
            raise Exception("'mode = 'training' or 'validation' or 'testing'")
        
    def lock(self):
          for j in ['W','dW']:
                for i in self.__buffer__[j].keys():
                      self.__buffer__[j+'0'][i][:] = self.__buffer__[j][i][:]

    
    """
    ADAPTED FROM
    % FEEC/Unicamp
    % 31/05/2017
    % function [Ew,dEw,eqm,CER,error_per_class] = process(X,S,Xv,Sv,w1,w2)
    % Output:  Ew: Squared error for the training dataset
    %          dEw: Gradient vector for the training dataset
    %          Ewv: Squared error for the validation dataset
    % Presentation of input-output patterns: batch mode
    % All neurons have bias
    """
    def process(self):
        self.eval_net('training')  
        x, y, S = self._partition_wrapper['training']
        w = self.W
        error = self.error
        dw = self.dW
        error[self.out_layer+1][:] = y[self.out_layer]-S
        dw[self.out_layer][:] = error[self.out_layer+1].T.dot(x[self.out_layer])
        for j in list.__reversed__(range(self.out_layer)):
              error[j+1][:] = _np.dot(error[j+2],w[j+1][:,:-1])*self.transf_fund(y[j])
              dw[j][:] = error[j+1].T.dot(x[j])

        Ew = 0.5*(error[self.out_layer+1].flatten('F').T.dot(error[self.out_layer+1].flatten('F'))) # total error ²
        dEw = _np.concatenate(([dw[i].flatten('F') for i in range(self.out_layer+1)])) # dw, gradient
        eqm = _np.sqrt((1./(self.Ntraining*self.n_out))*Ew) 
        return [Ew,dEw,eqm]
    
    """
    ADAPTED FROM
    # FEEC/Unicamp
    # 31/05/2017
    # function [s] = hprocess(X,S,w1,w2,p1,p2)
    # s = product H*p (computed exactly)
    """
    def hprocess(self):  
        self.eval_net('training')  
        x, y, S = self._partition_wrapper['training']
        rx, ry = [self.rX, self.rY]
        rerror, error = [self.rerror, self.error]
        w, dw, rw = [self.W, self.dW, self.rW]
        
        for j in range(self.out_layer+1):
              ry[j][:] = (x[j].dot(dw[j].T)+rx[j].dot(w[j].T))*self.transf_fund(y[j])
         
        rerror[self.out_layer+1][:] = ry[self.out_layer]
        rw[self.out_layer][:] = error[self.out_layer+1].T.dot(rx[self.out_layer])+rerror[self.out_layer+1].T.dot(x[self.out_layer])   
        for j in list.__reversed__(range(self.out_layer)):
              rerror[j+1][:] = (rerror[j+2].dot(w[j+1][:,:-1]) + error[j+2].dot(dw[j+1][:,:-1]))*self.transf_fund(y[j])+(error[j+2].dot(w[j+1][:,:-1]))*(-2*y[j]*ry[j])
              rw[j][:] = error[j+1].T.dot(rx[j])+rerror[j+1].T.dot(x[j])   

        rEw = _np.concatenate(([rw[i].flatten('F') for i in range(self.out_layer+1)])) # dw, gradient
        s = rEw
        return s
    
    def qmean2(self):
        w = self.W
        v = _np.concatenate(([w[i].flatten('F') for i in range(self.__buffer__['W'].__len__())]))  
        n_v = len(v)
        rms = _np.sqrt((v**2).sum()/n_v)
        return rms
    
    def m_norm(self,S):
        E = _np.sqrt((S**2).sum()/S.size);
        return E
    
    ### BASED ON 
    """
    # FEEC/Unicamp
    # 31/05/2017
    # nn1h_k_folds.m
    #
    """
    def train(self, threshold = 1.0e-5, n_itermax = 100, rate0 = 0.25, cut = 0.25):
        rate = rate0
        # Gradient norm: minimum value
        v_lambda = [0];
        blambda = 0;
        for k in self.__folders__.keys():
            self.set_fold(k)  #    
            folder = self.__folders__[k] # reference current folder in __folds__ dictionary
            self.reset_weights() # either assigns the current best weight for a loaded net or generate random ones
            t = _time.time() # tic toc
            
            w, rw, dw, w0, dw0 = [self.W, self.rW, self.dW, self.W0, self.dW0] # for easy reference herein on
            
            # initial conditions
            folder['rms_w'].append(self.qmean2()) # qmean2
            CER, CER_c = self.f_CER('validation') # Classificaiton error (absolute and per class)
            folder['error_per_class_v'].append(CER_c) 
            folder['CERv'].append(CER)
            Ew, dEw, eqm = self.process() 
            folder['eq'].append(Ew)
            folder['CER_min'] = min(folder['CERv'])
            
            # temporary variables for the following optimization
            iter_minor = 1
            p = -dEw; p_1 = p; r = -dEw;
            self.lock() # set W0 = W
            success = 1
            n_weights = sum([n.size for n in self.__buffer__['W'].values()])
            comp = []
            n_iter0 = int(folder['epoch'])
            
            print 'Classification Error: Validation {}, Training {}'.format(self.f_CER('validation')[0],self.f_CER('training')[0])
            
            #let if begin
            while self.m_norm(dEw) > threshold and folder['epoch']-n_iter0 < n_itermax: # convergence condition/criteria
                if success:
                    s = self.hprocess()
                    delta = p.T.dot(s)
                else:
                    print('[{},{},{}]'.format('Fail',folder['epoch'],delta))
                delta = delta+(v_lambda[-1]-blambda)*(p.T.dot(p))
                if delta <= 0: # Making delta a positive value
                    blambda = 2*(v_lambda[-1]-delta/(p.T.dot(p)));
                    delta = -delta+v_lambda[-1]*(p.T.dot(p));
                    v_lambda.append(blambda)
                mi = p.T.dot(r)
                alpha = mi/delta;
                for i in range(self.__buffer__['W'].__len__()):
                      w[i][:] = w0[i][:]-alpha*dw0[i][:]
                self.__haschanged__['training'] = True
                self.__haschanged__['validation'] = True
                [Ew1,dEw1,eqm1] = self.process()
                CER1, error_per_class1 = self.f_CER('validation')
                comp.append((Ew-Ew1)/(-dEw.T.dot(alpha*p)-0.5*((alpha**2)*delta)))
                #	In replacement to comp(it2) = 2*delta*(Ew-Ew1)/(mi^2); (is the same)
                if comp[-1] > 0:
                    print 'Classification Error (folder: {}, epoch: {}): Validation {}, Training {}'.format(k,folder['epoch'],CER1,self.f_CER('training')[0])
                    Ew = Ew1; folder['eq'].append(Ew)
                    if CER1 < folder['CER_min']:
                        folder['CER_min'] = CER1
                        for i in  self.__buffer__['W'].keys():
                              self.__buffer__['Wopt'][i][:] = self.__buffer__['W'][i][:]
                        folder['niter_v'] = folder['epoch']
                    CER = CER1 
                    folder['CERv'].append(CER)
                    dEw = dEw1
                    for i in range(self.stw.__len__()):
                          self.stw[i].append(self.m_norm(w0[i]-w[i]))
                    folder['rms_w'].append(self.qmean2())
                    folder['error_per_class_v'].append(error_per_class1)
                    eqm_fim = eqm1
                    folder['epoch'] += 1
                    r1 = r
                    r = -dEw
                    blambda = 0
                    success = 1
                    if (iter_minor == n_weights):
                        p_1 = p; p = r
                        iter_minor = 1
                    else:
                        iter_minor = iter_minor + 1;
                        beta = (r.T.dot(r)-r.T.dot(r1))/(r1.T.dot(r1)); # Polak-Ribiere (Luenberger, pp 253)
                        p_1 = p; p = r+beta*p
                        for i in self.__buffer__['dW'].keys():
                              dw[i][:] = dw[i][:]+beta*dw0[i][:]
                    if comp[-1] >= cut: 
                        v_lambda.append(rate*v_lambda[-1])
                        rate = rate*rate0
                    self.lock() # copy current W to W0
                    self.__haschanged__['training'] = True
                    self.__haschanged__['validation'] = True
                    folder['training_time'].append(_time.time()-t)
                else:
                    blambda = v_lambda[-1]
                    success = 0
                if comp[-1] < cut:
                    v_lambda.append(v_lambda[-1] + delta*(1-comp[-1])/(p_1.T.dot(p_1)))
                    rate = rate0
            print('Final mean squared error (training) = {} at iteration {}'.format(eqm_fim,folder['epoch']))
            print('Final CER (validation) = {} at iteration {}'.format(folder['CER_min'],folder['niter_v']))
            self.save_net('Networks/{}_MLP/{}_{}'.format(self.__name__,self.__name__,k))
            self.save_weights('Networks/{}_MLP/{}_vw_{}'.format(self.__name__,self.__name__,k))
            #plt.figure()
            #fig_CERv = plt.plot(self.CERv)
            #plt.gca().set_title('Evolution of the Classification Error Rate along training - folder = {:d}'.format(folder))
            #plt.gca().set_xlabel('Epochs')
            #plt.gca().set_ylabel('CER');
            #plt.figure()
            #fig_eq = plt.plot(self.eq)
            #plt.gca().set_title('Evolution of the quadratic error along training - folder = {:d}'.format(folder))
            #plt.gca().set_xlabel('Epochs')
            #plt.gca().set_ylabel('Quadratic error')
   
#ANN = MLP('Evol2',[10])
#ANN.load_data('xtraining.npz','xtesting.npz',[])
#ANN.init_weights()
#ANN.init_folds(5)
#ANN.train(threshold = 1.0e-5, n_itermax = 50,rate0 = 0.95,cut = 0.25)
#ANN.gen_k_folds('xtraining.npz',5)
#ANN.train()