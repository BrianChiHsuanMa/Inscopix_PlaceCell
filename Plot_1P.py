# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:09:36 2022

@author: SunLab
"""
#%% Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#%% Data directory
DataDir = 'C:\Files'
processed = os.path.join(DataDir, 'processed')
if not os.path.exists(processed):
    os.mkdir(processed)
#%% Define functions
def PlotTraces(data, cells: list, fs = 1, gap = 0, LR = 0, scalebar = 0, legend = 0):
    '''
    Plots the traces of the selected cells from a Suite2P cellset
    Inputs:
        data: relative path to the csv file that contains calcium data
        cells: the cell ids of selected cells
        fs: frame rate of the recorded movie
        gap: whether to connect the gaps in the recording time
        LR: whether the data is longitudinally registered
        scalebar: whether to plot a scalebar of the x and y axis
        legned: whether to plot a legend of each trace
    '''
    dftmp=pd.read_csv(os.path.join(DataDir,data))
    dftmp.drop(index=0,inplace=True)
    dftmp = dftmp.reset_index(drop=True) # Reset index
    dftmp = dftmp.astype(float) # Convert values to float
    if LR==1: # Find large gaps, mark with straight lines
        for i in dftmp.index[:-1]:
            if np.diff(dftmp.iloc[:,0])[i]>np.median(np.diff(dftmp.iloc[:,0]))+10000:
                dftmp.iloc[i+1:,0]-=(np.diff(dftmp.iloc[:,0])[i]-np.median(np.diff(dftmp.iloc[:,0])))
                plt.axvline(x=dftmp.iloc[i,0],ymin=0, ymax=len(dftmp.columns))
    if gap == 1: # Find and bridge gaps
        for i in dftmp.index[:-1]:
            if np.diff(dftmp.iloc[:,0])[i]>np.median(np.diff(dftmp.iloc[:,0]))+2:
                dftmp.iloc[i+1:,0]-=(np.diff(dftmp.iloc[:,0])[i]-np.median(np.diff(dftmp.iloc[:,0])))
    for i in dftmp.columns[1:]:
        dftmp[i] = stats.zscore(dftmp[i]) # Z-score activity
    fig, ax = plt.subplots()
    for n,i in enumerate(cells):
        ax.plot(dftmp.iloc[:,0],dftmp[i]+(n+1)*10,label=n+1)
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