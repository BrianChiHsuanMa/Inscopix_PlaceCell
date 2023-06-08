# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:22:33 2022

@author: SunLab
"""
#%% Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#%% Read Suite2P data
'''
Input the directory that contains the files created by Suite2P (F.npy, spks.npy, etc.)
'''
S2PDir = r'C:\Suite2P data'
F = np.load(os.path.join(S2PDir,'F.npy'), allow_pickle=True)
Fneu = np.load(os.path.join(S2PDir,'Fneu.npy'), allow_pickle=True)
spks = np.load(os.path.join(S2PDir,'spks.npy'), allow_pickle=True)
stat = np.load(os.path.join(S2PDir,'stat.npy'), allow_pickle=True)
ops =  np.load(os.path.join(S2PDir,'ops.npy'), allow_pickle=True)
ops = ops.item()
iscell = np.load(os.path.join(S2PDir,'iscell.npy'), allow_pickle=True)
# Create a directory to store generated figures
processed = os.path.join(S2PDir, 'processed')
if not os.path.exists(processed):
    os.mkdir(processed)
#%% Define functions
def PlotTraces2P(data, cells: list, fs = 10, scalebar = 0, legend = 0):
    '''
    Plots the traces of the selected cells from a Suite2P cellset
    Inputs:
        data: array that contains calcium activity of cells. Can be F or spks
        cells: the numbers of selected cells
        fs: frame rate of the recorded movie
        scalebar: whether to plot a scalebar of the x and y axis
        legned: whether to plot a legend of each trace
    '''
    dftmp=pd.DataFrame(data).transpose()
    fig, ax = plt.subplots()
    for i in dftmp.columns:
        dftmp[i] = stats.zscore(dftmp[i]) # Z-score activity
    for n,i in enumerate(cells):
        ax.plot(dftmp.index,dftmp[i]+(n+1)*10,label=n+1)
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
    fig.savefig(os.path.join(processed,'Traces.png'))
#%% Example workflow
PlotTraces2P(F,[1,2,3])