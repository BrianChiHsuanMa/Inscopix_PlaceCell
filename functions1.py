# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:09:36 2022

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
import os
import json
import random
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#%% Data directory
def create_savepath(data_dir):
    '''
    Creates a subfolder underneath the indicated directory to store output files

    Parameters
    ----------
    data_dir : string
        The indicated directory

    Returns
    -------
    None.

    '''
    processed = os.path.join(data_dir, 'processed')
    if not os.path.exists(processed):
        os.mkdir(processed)
#%%
def read_inputs(data_dir, use_denoise = True):
    files = os.listdir(data_dir)
    for file in files:
        if file.endswith('Denoise.csv') and use_denoise == True:
            df_traces = pd.read_csv(os.path.join(data_dir, file))
        elif file.endswith('Traces.csv') and use_denoise == False:
            df_traces = pd.read_csv(os.path.join(data_dir, file))
        elif file.endswith('Events.csv'):
            df_events = pd.read_csv(os.path.join(data_dir, file))
        elif file.endswith('GPIO.csv'):
            df_gpio = pd.read_csv(os.path.join(data_dir, file))
        elif 'shuffle' in file:
            df_motion = pd.read_csv(os.path.join(data_dir, file))
            
    df_gpio[['Time (s)',' Value']] = df_gpio[['Time (s)',' Value']].astype(float)
    
    for cell in df_traces.columns:
        if df_traces[cell][0] == ' rejected':
            df_traces.drop(columns=[cell], inplace=True)
    df_traces.drop(index=0,inplace=True)
    df_traces = df_traces.reset_index(drop=True).astype(float)
    df_traces.rename(columns = {' ':'Time (s)'}, inplace = True)
    
    df_events_expand = pd.DataFrame(data = df_traces['Time (s)'], columns = df_traces.columns)
    for cell in set(df_events[' Cell Name'].tolist()):
        df_current_cell = df_events.loc[df_events[' Cell Name'] == cell].copy()
        for spiketime in df_current_cell['Time (s)']:
            spikeval = float(df_current_cell.loc[df_current_cell['Time (s)'] == spiketime, ' Value'])
            actualtime = [time for time in df_events_expand['Time (s)'] if abs(time - spiketime) <= 0.05][0]
            df_events_expand.loc[df_events_expand['Time (s)'] == actualtime, cell] = spikeval
    
    df_motion.columns = df_motion.loc[0]+'_'+df_motion.loc[1]
    df_motion.drop(index=[0,1],inplace=True)
    df_motion.drop(columns = 'bodyparts_coords', inplace = True)
    df_motion.reset_index(drop=True,inplace=True)
    df_motion = df_motion.astype(float)
    
    return df_traces, df_events_expand, df_gpio, df_motion
#%%
def movement_analysis(df, df_ref, pthresh = 0.99, px2cm = 490/30, framerate = 10):
    df['head_x']=np.nan
    df['head_x'].loc[(df['neck_likelihood']>=pthresh)] = df['neck_x']
    df['head_x'].loc[(df['left ear_likelihood']>=pthresh) & (df['right ear_likelihood']>=pthresh)] = (df['left ear_x']+df['right ear_x'])/2
    df['head_y']=np.nan
    df['head_y'].loc[(df['neck_likelihood']>=pthresh)] = df['neck_y']
    df['head_y'].loc[(df['left ear_likelihood']>=pthresh) & (df['right ear_likelihood']>=pthresh)] = (df['left ear_y']+df['right ear_y'])/2
    
    df['head_x'].interpolate(inplace=True)
    df['head_y'].interpolate(inplace=True)
    
    newtime = list(np.linspace(df.index[0], df.index[-1], df_ref.shape[0]))
    df_new = pd.DataFrame(data = df_ref['Time (s)'], columns = ['Time (s)', 'head_x', 'head_y'])
    df_new['head_x'] = np.interp(newtime,df.index,df['head_x'])
    df_new['head_y'] = np.interp(newtime,df.index,df['head_y'])
    df_new['head_x'] = df_new['head_x'].rolling(window=5,center=True).mean()
    df_new['head_y'] = df_new['head_y'].rolling(window=5,center=True).mean()
    
    speed = [np.nan]
    for i in df_new.index[:-1]:
        currspd = (((df_new.loc[i+1,'head_x']-df_new.loc[i,'head_x'])**2 + 
                    (df_new.loc[i+1,'head_y']-df_new.loc[i,'head_y'])**2)**0.5) / px2cm * framerate
        speed.append(currspd)
    df_new['Speed']=speed
    
    return df_new
#%%
plotsession = CalciumData[plotday]['Traces']['session'+str(session)].copy()
indmid = int(len(plotsession.index)*0.5)
xlen = xlens[session]
xbound = xbounds[session]
xgridlen=xlen/30
dfzero = pd.DataFrame(0, columns = [i for i in range(30)],index = [i for i in range(30)])
dfoc = dfzero.copy()
for i in plotsession.index: # Calculate occupancy
    xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
    if xgrid < 0 or xgrid > 29: # Skip out of boundary data
        continue
    ygrid = plotsession.loc[i]['y']//16
    if ygrid < 0 or ygrid > 29:
        continue
    try:
        dfoc.at[ygrid,xgrid] += 1
    except KeyError:
        print(i)
dfoc.replace(0,np.nan,inplace=True) # Ignore unvisited bins
CellMaps['session' + str(session)]['Occupancys'] = dfoc.to_json(orient='columns')

for cell in plotevent.columns[1:]: # Calculate event map for each cell
    dfhm = dfzero.copy()
    indeventspre = plotevent[indst[session]:indet[session]+1].loc[plotevent[cell]> 0.0].index.tolist() # Grab index of events
    indevents = [i - indst[session] for i in indeventspre]
    for i in indevents:
        if plotsession['Speed'].loc[i] > 0.5: # Only count when speed > 0.5
            xgrid = (plotsession.loc[i]['x'] - xbound)//xgridlen
            if xgrid < 0 or xgrid > 29: # Skip out of boundary data
                continue
            ygrid = plotsession.loc[i]['y']//16
            if ygrid < 0 or ygrid > 29:
                continue
            dfhm.at[ygrid,xgrid] += plotevent.loc[i+indst[session]][cell]
    dfrt = dfhm/dfoc
    dfrt.replace(np.nan,0,inplace=True)
    dfrtgs=gaussian_filter(dfrt,sigma=2.5,truncate=2.0) # Gaussian smoothing with delta = 2.5 cm, 3*3 window
    dfrtgs=pd.DataFrame(dfrtgs/np.nanmax(dfrtgs.max())) # Normalize
    #dfrtgs.mask(dfrtgs<0.5,0,inplace=True) # Filter out bins with activity < 0.5 peak
    CellMaps['session' + str(session)][cell + 's'] = dfrtgs.to_json(orient='columns') # Save resulting map into JSON string for storage
#%%
create_savepath(r'C:\Users\owner\PlaceCell\SampleData')
read_inputs(r'C:\Users\owner\PlaceCell\SampleData')
df_motion_new = movement_analysis(df_motion,df_traces)
#%% Define functions
def PlotTraces(data, cells: list, gap = 0, scalebar = 0, legend = 0):
    dftmp=pd.read_csv(os.path.join(DataDir,data))

    dftmp.drop(index=0,inplace=True)
    dftmp = dftmp.reset_index(drop=True) # Reset index
    dftmp = dftmp.astype(float) # Convert values to float
    if gap == 1:
        for i in dftmp.index[:-1]: # Find gaps
            if np.diff(dftmp.iloc[:,0])[i]>np.median(np.diff(dftmp.iloc[:,0]))+2:
                dftmp.iloc[i+1:,0]-=(np.diff(dftmp.iloc[:,0])[i]-np.median(np.diff(dftmp.iloc[:,0])))
    for i in dftmp.columns[1:]: # Normalization
        #dftmp[i] = dftmp[i]/(dftmp[i].max()) # Normalize activity
        dftmp[i] = stats.zscore(dftmp[i]) # Z-score activity
    fig, ax = plt.subplots()
    for n,i in enumerate(cells):
        ax.plot(dftmp.iloc[:,0],dftmp[i]+(n+1)*10,label=n+1)
    if scalebar == 1:
        barx = AnchoredSizeBar(ax.transData, 10, '10 s', 'lower right',
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

def PlotTracesLR(data,gap: int):
    fig=plt.figure()
    ax = plt.subplot(111)
    dftmp=pd.read_csv(os.path.join(DataDir,data))
    dftmp.drop(index=0,inplace=True)
    dftmp = dftmp.reset_index(drop=True) # Reset index
    dftmp = dftmp.astype(float) # Convert values to float
    if gap==1:
        for i in dftmp.index[:-1]: # Find gaps
            if np.diff(dftmp.iloc[:,0])[i]>np.median(np.diff(dftmp.iloc[:,0]))+10000:
                dftmp.iloc[i+1:,0]-=(np.diff(dftmp.iloc[:,0])[i]-np.median(np.diff(dftmp.iloc[:,0])))
                plt.axvline(x=dftmp.iloc[i,0],ymin=0, ymax=len(dftmp.columns))
            elif np.diff(dftmp.iloc[:,0])[i]>np.median(np.diff(dftmp.iloc[:,0]))+2:
                dftmp.iloc[i+1:,0]-=(np.diff(dftmp.iloc[:,0])[i]-np.median(np.diff(dftmp.iloc[:,0])))
    for i in dftmp.columns[1:]: # Normalization
        dftmp[i] = dftmp[i]/(dftmp[i].max()) # Normalize activity
    for n,i in enumerate(dftmp.columns[1:]):
        ax.plot(dftmp.iloc[:,0],dftmp.iloc[:,i+1]+n,label=i)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),loc='upper right', bbox_to_anchor=(1.25, 1.05), fancybox=True, shadow=True)
    ax.get_yaxis().set_visible(False)
    
def PlotTracesMIRA(data,gap: int):
    fig=plt.figure()
    ax = plt.subplot(111)
    dftmp=pd.read_csv(os.path.join(DataDir,data))
    dftmp.drop(index=0,inplace=True)
    dftmp = dftmp.reset_index(drop=True) # Reset index
    dftmp = dftmp.astype(float) # Convert values to float
    if gap==1:
        for i in dftmp.index[:-1]: # Find gaps
            if np.diff(dftmp.iloc[:,0])[i]>np.median(np.diff(dftmp.iloc[:,0]))+2:
                dftmp.iloc[i+1:,0]-=(np.diff(dftmp.iloc[:,0])[i]-np.median(np.diff(dftmp.iloc[:,0])))
                plt.axvline(x=dftmp.iloc[i,0],ymin=0, ymax=len(dftmp.columns))
    for i in dftmp.columns[1:]: # Normalization
        dftmp[i] = dftmp[i]/(dftmp[i].max()) # Normalize activity
    for n,i in enumerate([11,45,59,60,54]):
        ax.plot(dftmp.iloc[:,0],dftmp.iloc[:,i+1]+n+1,label=n+1)
    handles, labels = ax.get_legend_handles_labels()
    bar = AnchoredSizeBar(ax.transData, 10, '10 s', 'lower right',
                      frameon=False, color='black')
    ax.add_artist(bar)
    ax.legend(reversed(handles), reversed(labels),loc='upper right', bbox_to_anchor=(1.25, 1.05), fancybox=True, shadow=True)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.set_ylim(0,6.5)
#%% Test
fig=plt.figure()
ax = plt.subplot(111)
bar = AnchoredSizeBar(ax.transData, 10, '10 s', 'lower right',
                  frameon=False, color='black')
bar2 = AnchoredSizeBar(ax.transData, 0.1, '10 sd', 'lower right',
                  pad = 1, frameon=True, size_vertical = 10, label_top = True,
                  color='black')
ax.add_artist(bar)
ax.add_artist(bar2)