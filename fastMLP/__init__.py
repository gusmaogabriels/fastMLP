# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = {'Gabriel S. Gusmao' : 'gusmaogabriels@gmail.com'}
__version__ = '1.0a'

"""
By Gabriel S. Gusmão (Gabriel Sabença Gusmão)
Sep, 2017
    fastMLP version 1.0a
    ~~~~
    "Fast batch MLP (Multi-layer Perceptron) algorithm based on numpy memmap"
    :copyright: (c) 2017 Gabriel S. Gusmão
    :license: GPU, see LICENSE for more details.
"""

import numpy as np
import os
import shutil
import time
import fastMLP
import linClassifier

__all__ = ['np','__author__','__version__']
