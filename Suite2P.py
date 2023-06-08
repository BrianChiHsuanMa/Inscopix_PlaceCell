# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:22:33 2022

@author: SunLab
"""
#%% Importing packages
import pandas as pd
from pathlib import Path  # to work with dir
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statistics as st
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy import stats
import json
import random
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#%% Data directory
S2PDir = r'D:\2P data\ISX-YSN1-00039\Test 5 12162022\TSeries-12162022-1110-121\Ch2Stack\suite2p\plane0'
#%% Define functions
def readdata(Dir):
    F = np.load(os.path.join(Dir,'F.npy'), allow_pickle=True)
    Fneu = np.load(os.path.join(Dir,'Fneu.npy'), allow_pickle=True)
    spks = np.load(os.path.join(Dir,'spks.npy'), allow_pickle=True)
    stat = np.load(os.path.join(Dir,'stat.npy'), allow_pickle=True)
    ops =  np.load(os.path.join(Dir,'ops.npy'), allow_pickle=True)
    ops = ops.item()
    iscell = np.load(os.path.join(Dir,'iscell.npy'), allow_pickle=True)
    F_iscell=F[iscell[:,0]==1]
    S_iscell=spks[iscell[:,0]==1]
    
def PlotTraces2P(data, cells: list, fs = 10, scalebar = 0, legend = 0):
    dftmp=pd.DataFrame(data).transpose()
    fig, ax = plt.subplots()
    for i in dftmp.columns: # Normalization
        dftmp[i] = stats.zscore(dftmp[i]) # Z-score activity
    for n,i in enumerate(cells):
        ax.plot(dftmp.index,dftmp[i]+(n+1)*10,label=n+1)
    bar = AnchoredSizeBar(ax.transData, 100, '10 s', 'lower right',
                      frameon=False, color='black')
    if scalebar == 1:
        barx = AnchoredSizeBar(ax.transData, 10*fs, '10 s', 'lower right',
                          frameon=False, 
                          bbox_transform=ax.transAxes, bbox_to_anchor=(1.05,0),
                          color='black')
        bary = AnchoredSizeBar(ax.transData, 0.1, '10 sd', 'lower right',
                          frameon=False, size_vertical = 10, label_top = True,
                          bbox_transform=ax.transAxes, bbox_to_anchor=(1.1,0.05),
                          color='black')
        ax.add_artist(barx)
        ax.add_artist(bary)
    if legend == 1:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper right', bbox_to_anchor=(1.25,1), fancybox=True, shadow=True)
    ax.axis('off')
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_ylim(0,len(cells)*10+15)