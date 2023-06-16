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
savepath = create_savepath(r'C:\Users\owner\PlaceCell\SampleData')
df_traces, df_events, df_gpio, processed, n_sessions = read_inputs(
    r'C:\Users\owner\PlaceCell\SampleData')
traces_start, traces_end, vid_start, vid_end = process_GPIO(df_gpio)
process_calcium(df_traces, df_events, processed, n_sessions)
for s in range(n_sessions):
    df_move = movement_analysis(processed['df_vid' + str(s)],
                                processed['df_traces' + str(s)])
    processed['df_move' + str(s)] = df_move.copy()

x_lens = [490, 490, 490, 490, 490]
x_bounds = [60, 75, 60, 80, 70]
y_lens = [480,480,480,480,480]
y_bounds = [0, 0, 0, 0, 0]
cellmaps_all = {}
cellmaps_first = {}
cellmaps_second = {}
for s in range(n_sessions):
    df_events_curr = processed['df_events' + str(s)]
    df_move_curr = processed['df_move' + str(s)]
    cellmaps_all_curr = calculate_cellmap(df_events_curr, df_move_curr,
                                          x_bounds[s], x_lens[s], y_bounds[s],
                                          y_lens[s], half = 'all')
    cellmaps_all['session' + str(s)] = cellmaps_all_curr
    cellmaps_first_curr = calculate_cellmap(df_events_curr, df_move_curr,
                                            x_bounds[s], x_lens[s], y_bounds[s],
                                            y_lens[s], half = 'first')
    cellmaps_first['session' + str(s)] = cellmaps_first_curr
    cellmaps_second_curr = calculate_cellmap(df_events_curr, df_move_curr,
                                             x_bounds[s], x_lens[s], y_bounds[s],
                                             y_lens[s], half = 'second')
    cellmaps_second['session' + str(s)] = cellmaps_second_curr
    
for s in cellmaps_first.keys():
    df_placecell = placecell_identification(cellmaps_first[s], cellmaps_second[s])
    df_placecell.to_csv(os.path.join(savepath, s + ' Place Cell Identification.csv'))
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
    savepath = os.path.join(data_dir, 'processed')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    return savepath

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
    processed = {}
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
            processed['df_vid' + str(n_sessions)] = df_currvid.copy()
            n_sessions = n_sessions + 1
            
    return df_traces, df_events, df_gpio, processed, n_sessions
      
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

def process_calcium(df_traces, df_events, processed, n_sessions):
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
        processed['df_traces' + str(s)] = df_traces_curr
        df_events_curr = df_events_expand[ind_start:ind_end+1].copy().reset_index(drop=True)
        df_events_curr['Time (s)'] = df_events_curr['Time (s)'] - df_events_curr['Time (s)'][0]
        processed['df_events' + str(s)] = df_events_curr

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

def calculate_cellmap(df_events, df_move, x_bound, x_len,
                      y_bound, y_len, n_grid = 30, half = 'all'):
    
    cellmaps = {}
    grid_x, grid_y = x_len/n_grid, y_len/n_grid
    df_zero = pd.DataFrame(0, columns = [i for i in range(n_grid)],
                           index = [i for i in range(n_grid)])
    df_occ = df_zero.copy()
    ind_mid = int(df_events.shape[0]/2)
    
    df_grid = pd.DataFrame({'x': (df_move['head_x'] - x_bound)//grid_x,
                           'y': (df_move['head_y'] - y_bound)//grid_y})
    ind_occ = df_grid.loc[df_grid['x'].between(0, 29)].loc[df_grid['y'].between(0, 29)].index
    if half == 'first':
        for i in ind_occ.intersection(df_grid.index[:ind_mid]):
            df_occ.at[df_grid.loc[i, 'y'],df_grid.loc[i, 'x']] += 1
    elif half == 'second':
        for i in ind_occ.intersection(df_grid.index[ind_mid:]):
            df_occ.at[df_grid.loc[i, 'y'],df_grid.loc[i, 'x']] += 1
    elif half == 'all':
        for i in ind_occ:
            df_occ.at[df_grid.loc[i, 'y'],df_grid.loc[i, 'x']] += 1

    df_occ.replace(0, np.nan, inplace=True) # Ignore unvisited bins
        
    for cell in df_events.columns[1:]: # Calculate event map for each cell
        df_spk = df_zero.copy()
        ind_spk = df_events.loc[df_events[cell] > 0].index
        ind_run = df_move.loc[df_move['Speed'] >= 0.5].index
        if half == 'first':
            for i in ind_spk.intersection(ind_run).intersection(ind_occ).intersection(df_grid.index[:ind_mid]):
                df_spk.at[df_grid.loc[i, 'y'],df_grid.loc[i, 'x']] += df_events.loc[i, cell]
        elif half == 'second':
            for i in ind_spk.intersection(ind_run).intersection(ind_occ).intersection(df_grid.index[ind_mid:]):
                df_spk.at[df_grid.loc[i, 'y'],df_grid.loc[i, 'x']] += df_events.loc[i, cell]
        elif half == 'all':
            for i in ind_spk.intersection(ind_run).intersection(ind_occ):
                df_spk.at[df_grid.loc[i, 'y'],df_grid.loc[i, 'x']] += df_events.loc[i, cell]

        df_rate = df_spk / df_occ
        df_rate.replace(np.nan,0,inplace=True)
        df_rate = gaussian_filter(df_rate, sigma = 2.5, truncate=2.0) # Gaussian smoothing with delta = 2.5 cm, 3*3 window
        df_rate = pd.DataFrame(df_rate/np.nanmax(df_rate.max()))
        #CellMaps['session' + str(session)][cell + 's1'] = dfrtgs.to_json(orient='columns')
        cellmaps[cell] = df_rate
    
    return cellmaps
#%% Place cell identification with stability method
def placecell_identification(cellmaps_first_curr, cellmaps_second_curr):
    maps1 = [i for i in cellmaps_first_curr.keys()]
    maps2 = [i for i in cellmaps_second_curr.keys()]
    corr_scores = []
    thresholds = []
    is_placecell = []
    
    for k1 in maps1:
        map1 = cellmaps_first_curr[k1].to_numpy().flatten()
        map2 = cellmaps_second_curr[k1].to_numpy().flatten()
        corr_score = np.corrcoef(map1, map2)[0][1]
        corr_scores.append(corr_score)
        
        random_score = []
        for n in range(500):
            k2 = maps2[random.randint(0,len(maps2)-1)]
            map2 = cellmaps_second_curr[k2].to_numpy().flatten()
            random_score.append(np.corrcoef(map1, map2)[0][1])
        threshold = np.nanpercentile(random_score, 95)
        thresholds.append(threshold)
    
        if corr_score > threshold:
            is_placecell.append('Accepted')
        else:
            is_placecell.append('Rejected')
    
    df_placecell = pd.DataFrame([corr_scores, thresholds, is_placecell],
                                columns = maps1, index = ['Score', 'P95', 'Is place cell?'])
    
    return df_placecell    
#%%
dfcorr=pd.DataFrame(columns=[i[:4] for i in CellMaps['session0'].keys() if i.endswith('s1') and i!='Occupancys1'])
dfcorr['row']=[]
for n,s in enumerate([k for k in CellMaps.keys() if k != 'day']):
    dfcorr.loc[n*3,'row']='Threshold '+s
    dfcorr.loc[n*3+1,'row']='Correlation '+s
    dfcorr.loc[n*3+2,'row']='Status '+s
    maps1=[i for i in CellMaps[s].keys() if i.endswith('s1') and i!='Occupancys1']
    maps2=[i for i in CellMaps[s].keys() if i.endswith('s2') and i!='Occupancys2']
    for k in maps1:
        corrlist=[]
        map1 = pd.read_json(CellMaps[s][k]).to_numpy().flatten()
        for i in range(500):
            k2 = maps2[random.randint(0,len(maps2)-1)]
            map2 = pd.read_json(CellMaps[s][k2]).to_numpy().flatten()
            corrlist.append(np.corrcoef(map1,map2)[0][1])
        p95 = np.nanpercentile(corrlist,95)
        dfcorr.loc[n*3,k[:4]]=p95
        k2 = k[:-1]+'2'
        map2 = pd.read_json(CellMaps[s][k2]).to_numpy().flatten()
        dfcorr.loc[n*3+1,k[:4]]=np.corrcoef(map1,map2)[0][1]
        if np.corrcoef(map1,map2)[0][1]>p95:
            dfcorr.loc[n*3+2,k[:4]]='Accepted'
        else:
            dfcorr.loc[n*3+2,k[:4]]='Rejected'
dfcorr.set_index('row',inplace=True)
dfcorr.to_csv(CellMaps['day']+' Place Cell Identification.csv')