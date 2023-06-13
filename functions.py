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
#%% run block
df_traces, df_events, df_gpio, sessions, n_sessions = read_inputs(
    r'C:\Users\owner\PlaceCell\SampleData')
traces_start, traces_end, vid_start, vid_end = process_GPIO(df_gpio)
process_calcium(df_traces, df_events, sessions, n_sessions)
for s in range(n_sessions):
    df_move = movement_analysis(sessions['df_vid' + str(s)],
                                sessions['df_traces' + str(s)])
    sessions['df_move' + str(s)] = df_move.copy()
#%%
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

def read_inputs(data_dir, use_denoise = True):
    '''
    Read all csv files in directory into Dataframes

    Parameters
    ----------
    data_dir : path
        Path to dir where the files are saved.
    use_denoise : boolean, optional
        Whether to use the denoised traces. The default is True.

    Returns
    -------
    df_traces : Dataframe
        DESCRIPTION.
    df_events : Dataframe
        DESCRIPTION.
    df_gpio : Dataframe
        DESCRIPTION.
    sessions : dictionary
        DESCRIPTION.
    n_sessions : integer
        DESCRIPTION.

    '''
    files = os.listdir(data_dir)
    sessions = {}
    n_sessions = 0
    for file in files:
        if file.endswith('Denoise.csv') and use_denoise == True:
            df_traces = pd.read_csv(os.path.join(data_dir, file))
        elif file.endswith('Traces.csv') and use_denoise == False:
            df_traces = pd.read_csv(os.path.join(data_dir, file))
        elif file.endswith('Events.csv'):
            df_events = pd.read_csv(os.path.join(data_dir, file))
        elif file.endswith('GPIO.csv'):
            df_gpio = pd.read_csv(os.path.join(data_dir, file))
        elif 'shuffle' in file: # Reads and cleans motion data
            df_currvid = pd.read_csv(os.path.join(data_dir, file))
            df_currvid.columns = df_currvid.loc[0]+'_'+df_currvid.loc[1]
            df_currvid.drop(index=[0,1],inplace=True)
            df_currvid.drop(columns = 'bodyparts_coords', inplace = True)
            df_currvid.reset_index(drop=True,inplace=True)
            df_currvid = df_currvid.astype(float)
            sessions['df_vid' + str(n_sessions)] = df_currvid.copy()
            n_sessions = n_sessions + 1
            
    return df_traces, df_events, df_gpio, sessions, n_sessions
#%%         
def process_GPIO(df_gpio):
    df_gpio[['Time (s)',' Value']] = df_gpio[['Time (s)',' Value']].astype(float)

    # Find calcium recording sessions start and end time
    df_gpio_traces = df_gpio.loc[df_gpio[' Channel Name']==' EX-LED']
    traces_delay = df_gpio_traces.iloc[1]['Time (s)']
    traces_start = df_gpio_traces.loc[df_gpio_traces[' Value'] > 0]['Time (s)'] - traces_delay
    traces_end = df_gpio_traces.loc[traces_start.index + 1]['Time (s)'] - traces_delay
    traces_start.reset_index(inplace = True, drop = True)
    traces_end.reset_index(inplace = True, drop = True)
    
    # Find behavioural video start and end time
    df_gpio_vid = df_gpio.loc[df_gpio[' Channel Name']==' BNC Trigger Input']
    #motion_delay = df_gpio_motion.iloc[1]['Time (s)']
    vid_start = df_gpio_vid.loc[df_gpio_vid[' Value'] > 0]['Time (s)'] - traces_delay
    vid_end = df_gpio_vid.loc[vid_start.index + 1]['Time (s)'] - traces_delay
    vid_start.reset_index(inplace = True, drop = True)
    vid_end.reset_index(inplace = True, drop = True)
    
    # Perform several tests to validate synchronization
    if vid_start.shape[0] != n_sessions:
        print('Wrong number of motion sessions extracted from GPIO')
    if traces_start.shape[0] != n_sessions:
        print('Wrong number of calcium sessions extracted from GPIO')
    if any(abs(vid_start - traces_start) > 0.1):
        print('Motion and calcium data are misaligned')
    
    return traces_start, traces_end, vid_start, vid_end
#%%
def process_calcium(df_traces, df_events, sessions, n_sessions):
    for cell in df_traces.columns:
        if df_traces[cell][0] == ' rejected':
            df_traces.drop(columns=[cell], inplace=True)
    df_traces.drop(index=0,inplace=True)
    df_traces = df_traces.reset_index(drop=True).astype(float)
    df_traces.rename(columns = {' ':'Time (s)'}, inplace = True)
    
    df_events_expand = pd.DataFrame(data = df_traces['Time (s)'],
                                    columns = df_traces.columns)
    for cell in set(df_events[' Cell Name'].tolist()):
        df_current_cell = df_events.loc[df_events[' Cell Name'] == cell].copy()
        for spiketime in df_current_cell['Time (s)']:
            spikeval = float(df_current_cell.loc[df_current_cell['Time (s)'] == spiketime, ' Value'])
            ind_spike = (df_traces['Time (s)'] - spiketime).abs().argsort()[0]
            df_events_expand.loc[ind_spike, cell] = spikeval
    
    for s in range(n_sessions):
        ind_start = (df_traces['Time (s)'] - traces_start[s]).abs().argsort()[0]
        ind_end = (df_traces['Time (s)'] - traces_end[s]).abs().argsort()[0]
        df_traces_curr = df_traces[ind_start:ind_end+1].copy().reset_index(drop=True)
        df_traces_curr['Time (s)'] = df_traces_curr['Time (s)'] - df_traces_curr['Time (s)'][0]
        sessions['df_traces' + str(s)] = df_traces_curr
        df_events_curr = df_events_expand[ind_start:ind_end+1].copy().reset_index(drop=True)
        df_events_curr['Time (s)'] = df_events_curr['Time (s)'] - df_events_curr['Time (s)'][0]
        sessions['df_events' + str(s)] = df_events_curr
#%%
def movement_analysis(df_vid, df_ref, pthresh = 0.99, px2cm = 490/30, framerate = 10):
    df_vid['head_x']=np.nan
    df_vid['head_x'].loc[(df_vid['neck_likelihood']>=pthresh)] = df_vid['neck_x']
    df_vid['head_x'].loc[(df_vid['left ear_likelihood']>=pthresh) &
                     (df_vid['right ear_likelihood']>=pthresh)] = (df_vid['left ear_x'] + df_vid['right ear_x'])/2
    df_vid['head_y']=np.nan
    df_vid['head_y'].loc[(df_vid['neck_likelihood']>=pthresh)] = df_vid['neck_y']
    df_vid['head_y'].loc[(df_vid['left ear_likelihood']>=pthresh) & 
                     (df_vid['right ear_likelihood']>=pthresh)] = (df_vid['left ear_y'] + df_vid['right ear_y'])/2
    
    df_vid['head_x'].interpolate(inplace=True)
    df_vid['head_y'].interpolate(inplace=True)
    
    newtime = list(np.linspace(df_vid.index[0], df_vid.index[-1], df_ref.shape[0]))
    df_new = pd.DataFrame(data = df_ref['Time (s)'], columns = ['Time (s)', 'head_x', 'head_y'])
    df_new['head_x'] = np.interp(newtime,df_vid.index,df_vid['head_x'])
    df_new['head_y'] = np.interp(newtime,df_vid.index,df_vid['head_y'])
    df_new['head_x'] = df_new['head_x'].rolling(window=5,center=True).mean()
    df_new['head_y'] = df_new['head_y'].rolling(window=5,center=True).mean()
    
    speed = [np.nan]
    for i in df_new.index[:-1]:
        currspd = (((df_new.loc[i+1,'head_x']-df_new.loc[i,'head_x'])**2 + 
                    (df_new.loc[i+1,'head_y']-df_new.loc[i,'head_y'])**2)**0.5) / px2cm * framerate
        speed.append(currspd)
    df_new['Speed']=speed
    
    return df_new