# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:08:59 2017

@data_processer author: Gabriel S. Gusmão, gusmaogabriels@gmail.com
"""

import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
    
def data_processer(iterator,n_images,n_bits,n_classes,addbias=True):
    images = np.ones([n_images,n_bits+1*addbias],np.float32)
    classes = np.zeros([n_images,n_classes],np.float32)
    classif = np.ndarray(n_images,np.float64)
    count = 0
    for i in iterator:
        if addbias:
            images[count,:-1] = i[1].flatten('F')/255.
        else:
            images[count,:] = i[1].flatten('F')/255.
        classes[count,i[0]] = 1.
        classif[count] = i[0]
        count += 1
    return images, classes, classif

def data_dump():
    X_t0, S_t0, l_t0  = data_processer(read(dataset='training'),60000,28*28,10,addbias=False) # carregamento dos dados
    np.savez_compressed('xtraining',X=X_t0,S=S_t0)
    X_ts, S_ts, l_ts  = data_processer(read(dataset='testing'),10000,28*28,10,addbias=False) # validação
    np.savez_compressed('xtesting',X=X_ts,S=S_ts)
    