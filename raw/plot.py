# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 18:58:10 2017

@author: GABRIS46
"""

import matplotlib.pyplot as plt

import locale
# Set to German locale to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, "US")

plt.rcParams['axes.formatter.use_locale'] = True

import pandas as pd
import seaborn as sns
import numpy as np
import os
import time
import uuid
import gc
from scipy import stats

sns.set_style('white')


font = 'Segoe UI'
plt.rc('font',family=font)
plt.rc('mathtext',fontset='custom')
plt.rc('mathtext',rm=font)    
plt.rc('mathtext',it='{}:italic'.format(font))
plt.rc('mathtext',bf='{}:bold'.format(font))
plt.rc('mathtext',default='regular')
fs=14
plt.rc('font', size=fs)          # controls default text sizes
plt.rc('axes', titlesize=fs)     # fontsize of the axes title#
plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
plt.rc('legend', fontsize=fs)    # legend fontsize
plt.rc('figure', titlesize=fs)  # fontsize o

plt.rc('text.latex',unicode=True)

