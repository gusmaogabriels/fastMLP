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
import networkx as nx

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

G = nx.Graph()

nlayers = [4,6,5,3]

edges = []
positions = {}
labels_n = {}
labels_b = {}
labels_e = {}

for layer in range(len(nlayers)-1):
      pos0 = 0.75
      delta = 0.65/(nlayers[layer])
      labels_e[layer] = {}
      if layer  == len(nlayers)-2:
            flag = True
      else:
            flag = False            
      for node_a in range(nlayers[layer]):
            labels_e[layer][node_a] = {'position':{},'edge':{}}
            for node_b in range(nlayers[layer+1]-1+flag*1):
                  edges += [[str(layer)+'_'+str(node_a),str(layer+1)+'_'+str(node_b)]]
                  labels_e[layer][node_a]['edge'][str(layer)+'_'+str(node_a),str(layer+1)+'_'+str(node_b)] = '$w^{0}_{{{1}}}$'.format(layer,str(node_a)+str(node_b))
            labels_e[layer][node_a]['position'] = pos0
            pos0 -= delta
            
pos0x = 0
color0 = 0.25
for layer in range(len(nlayers)):
      k = nlayers[layer]
      delta = 1./(k+1)
      pos = -(delta*k)/2.
      for node in range(nlayers[layer]):
            if node == nlayers[layer]-1 and layer<len(nlayers)-1:
                  labels_b[str(layer)+'_'+str(node)] = '$b_{}$'.format(layer,node)
            else:
                  labels_n[str(layer)+'_'+str(node)] = 'n$_{0}$$_{1}$'.format(layer,node)
            positions[str(layer)+'_'+str(node)] = [pos0x,pos]
            pos += delta*1.5
      pos0x += 10
      color0 *= 1.5
      
#G.add_edges_from(edges)      
c = ['gold','silver','silver','red']
for layer in range(len(nlayers)):
      for node in range(nlayers[layer]):
            if node == nlayers[layer]-1 and layer<len(nlayers)-1:
                  nx.draw_networkx_nodes(G,positions,nodelist=[str(layer)+'_'+str(node)],node_color='black',node_size=400,alpha=0.8,node_shape='s')
            else:
                  nx.draw_networkx_nodes(G,positions,nodelist=[str(layer)+'_'+str(node)],node_color=c[layer],node_size=450,alpha=1,linewidths=1,node_shape='o')     
plt.gcf().set_size_inches(10,10/1.6)
nx.draw_networkx_labels(G,positions,labels_n,font_color='black',font_size=12)
nx.draw_networkx_labels(G,positions,labels_b,font_color='white',font_size=12)
nx.draw_networkx_edges(G,positions,edges,linewidth=0.5,alpha=0.5)

pos = [0.8,0.75,0.8]
for l in labels_e.keys():
      for node in labels_e[l].values(): 
            nx.draw_networkx_edge_labels(G,positions,edge_labels=node['edge'],label_pos=node['position'],bbox={'boxstyle':'square','facecolor':'white', 'alpha':0.85, 'pad':0,'lw':0},font_size=8)

bottom = -0.6
top = 0.7
plt.gca().text(0,bottom,'Input Layer',fontsize=12,ha='center')
plt.gca().text(5,top,'Input Layer\nWeights',fontsize=12,ha='center')
plt.gca().text(10,bottom,'Hidden Layer 1',fontsize=12,ha='center')
plt.gca().text(15,top,'Hidden Layer 1\nWeights',fontsize=12,ha='center')
plt.gca().text(20,bottom,'Hidden Layer 2',fontsize=12,ha='center')
plt.gca().text(25,top,'Hidden Layer 2\nWeights',fontsize=12,ha='center')
plt.gca().text(30,bottom,'Output Layer',fontsize=12,ha='center')
plt.gca().text(-1.25,0.509,'bias',fontsize=12,ha='center',va='center',rotation=90)
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.axis('off')
plt.tight_layout(rect=[0.05,0.05,0.95,0.95])