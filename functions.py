#%% Importing packages
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import os
import random
#%%
def create_savepath(data_dir):
    """
    Creates a subfolder underneath the indicated directory to store outputs.

    Parameters
    ----------
    data_dir : path
        Where the data are stored.

    Returns
    -------
    savepath : path
        Directory to store output files.

    """
    savepath = os.path.join(data_dir, 'processed')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    return savepath

def read_inputs(data_dir, use_denoise = True):
    '''
    Read all csv files in directory into Dataframes.

    Parameters
    ----------
    data_dir : path
        Where the data are stored.
    use_denoise : boolean, optional
        Whether to use the denoised traces. The default is True.

    Returns
    -------
    df_traces : DataFrame
        DataFrame containing calcium activity information.
    df_events : DataFrame
        DataFrame containing spiking information.
    df_gpio : DataFrame
        DataFrame containing temporal information.
    processed : dictionary
        Dictionary containing processed data. At this stage it will store the
        processed motion detection data, stored in DataFrames.
    n_sessions : integer
        Number of video recording sessions.

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
      
def process_GPIO(df_gpio, n_sessions):
    """
    Extract temporal information from the GPIO file.

    Parameters
    ----------
    df_gpio : DataFrame
        DataFrame containing temporal information.
    n_sessions : integer
        Number of video recording sessions.

    Returns
    -------
    traces_start : Series
        Starting time of all calcium recording sessions.
    traces_end : Series
        Ending time of all calcium recording sessions.
    vid_start : Series
        Starting time of all video recording sessions.
    vid_end : Series
        Ending time of all calcium recording sessions.

    """
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

def process_calcium(df_traces, df_events, processed, n_sessions, traces_start,
                    traces_end):
    """
    Split calcium and spiking data into separate sessions, and store the 
    outputs in the "processed" dictionary.

    Parameters
    ----------
    df_traces : DataFrame
        DataFrame containing calcium activity information.
    df_events : TYPE
        DataFrame containing spiking information.
    processed : TYPE
        Dictionary where the processed data will be stored.
    n_sessions : TYPE
        Number of sessions.
    traces_start : TYPE
        Starting time of all calcium recording sessions.
    traces_end : TYPE
        Ending time of all calcium recording sessions.

    Returns
    -------
    None.

    """
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
        df_traces_curr['Time (s)'] = (df_traces_curr['Time (s)'] -
                                      df_traces_curr['Time (s)'][0])
        processed['df_traces' + str(s)] = df_traces_curr
        df_events_curr = df_events_expand[ind_start:ind_end+1].copy().reset_index(drop=True)
        df_events_curr['Time (s)'] = (df_events_curr['Time (s)'] -
                                      df_events_curr['Time (s)'][0])
        processed['df_events' + str(s)] = df_events_curr

def movement_analysis(df_vid, df_ref, pthresh = 0.99, px2cm = 490/30, framerate = 10):
    """
    Calculate actual position and speed from a motion detection data.

    Parameters
    ----------
    df_vid : DataFrame
        DataFrame containing motion detection data.
    df_ref : DataFrame
        DataFrame whose temporal information will be used as reference.
        Ideally this should be the calcium data from the same session.
    pthresh : float, optional
        Threshold to filter motion detection results. The default is 0.99.
    px2cm : float, optional
        Conversion ratio for pixel to cm. The default is 490/30.
    framerate : int, optional
        Frame rate in which the data used as reference (df_ref) is captured.
        The default is 10.

    Returns
    -------
    df_new : DataFrame
        DataFrame containing the processed motion data.

    """
    df_vid['head_x']=np.nan
    df_vid['head_x'].loc[(df_vid['neck_likelihood'] >= pthresh)] = df_vid['neck_x']
    df_vid['head_x'].loc[(df_vid['left ear_likelihood'] >= pthresh) &
                     (df_vid['right ear_likelihood'] >= pthresh)] = (df_vid['left ear_x'] + df_vid['right ear_x'])/2
    df_vid['head_y']=np.nan
    df_vid['head_y'].loc[(df_vid['neck_likelihood'] >= pthresh)] = df_vid['neck_y']
    df_vid['head_y'].loc[(df_vid['left ear_likelihood'] >= pthresh) & 
                     (df_vid['right ear_likelihood'] >= pthresh)] = (df_vid['left ear_y'] + df_vid['right ear_y'])/2
    
    df_vid['head_x'].interpolate(inplace=True)
    df_vid['head_y'].interpolate(inplace=True)
    
    newtime = list(np.linspace(df_vid.index[0], df_vid.index[-1], df_ref.shape[0]))
    df_new = pd.DataFrame(data = df_ref['Time (s)'],
                          columns = ['Time (s)', 'head_x', 'head_y'])
    df_new['head_x'] = np.interp(newtime, df_vid.index, df_vid['head_x'])
    df_new['head_y'] = np.interp(newtime, df_vid.index, df_vid['head_y'])
    df_new['head_x'] = df_new['head_x'].rolling(window = 5, center = True).mean()
    df_new['head_y'] = df_new['head_y'].rolling(window = 5, center = True).mean()
    
    speed = [np.nan]
    for i in df_new.index[:-1]:
        currspd = (((df_new.loc[i+1, 'head_x'] - df_new.loc[i, 'head_x'])**2 + 
                    (df_new.loc[i+1, 'head_y'] - df_new.loc[i, 'head_y'])**2)**0.5)
        currspd = currspd / (px2cm * framerate)
        speed.append(currspd)
    df_new['Speed']=speed
    
    return df_new

def calculate_placemap(df_events, df_move, x_bound = 0, x_len = 480,
                       y_bound = 0, y_len = 480, n_grid = 30, half = 'all'):
    """
    Calculate place fields based on spiking and motion data.

    Parameters
    ----------
    df_events : DataFrame
        DataFrame containing spiking information. Requires the processed form.
    df_move : DataFrame
        DataFrame containing motion data.
    x_bound : int
        Left boundary of the arena in the captured video, in pixels.
        The default is 0.
    x_len : int
        Length of the arena on the x axis, in pixels. The default is 480.
    y_bound : TYPE
        Upper bondary of the arena in the captured video, in pixels.
        The default is 0.
    y_len : TYPE
        Length of the arena on the y axis, in pixels. The default is 480.
    n_grid : int, optional
        Number of grids to seperate the arena into, on each axis.
        The default is 30, producing a 30x30 grid.
    half : str, optional
        Which half of the session to be used. The default is 'all'.
        Accepted values:
            'all': The entire session.
            'first': The first half.
            'second': The second half.

    Returns
    -------
    placemaps : dict
        Dictonary containing calculated placemaps for each cell.

    """
    placemaps = {}
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
        
    for cell in df_events.columns[1:]:
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
        df_rate = gaussian_filter(df_rate, sigma = 2.5, truncate=2.0) # Gaussian smoothing with delta = 2.5, 3*3 window
        df_rate = pd.DataFrame(df_rate/np.nanmax(df_rate.max()))
        placemaps[cell] = df_rate
    
    return placemaps

def placecell_identification(placemaps_first_curr, placemaps_second_curr):
    """
    Perform place cell identification based on stability method.

    Parameters
    ----------
    placemaps_first_curr : dict
        Dictionary containing place maps of each cell, calculated from the
        first half of a session.
    placemaps_second_curr : dict
        Dictionary containing place maps of each cell, calculated from the
        second half of a session.

    Returns
    -------
    df_placecell : DataFrame
        DataFrame containing place cell identification results
        (correlation score, threshold, place cell status).

    """
    maps1 = [i for i in placemaps_first_curr.keys()]
    maps2 = [i for i in placemaps_second_curr.keys()]
    corr_scores = []
    thresholds = []
    is_placecell = []
    
    for k1 in maps1:
        map1 = placemaps_first_curr[k1].to_numpy().flatten()
        map2 = placemaps_second_curr[k1].to_numpy().flatten()
        corr_score = np.corrcoef(map1, map2)[0][1]
        corr_scores.append(corr_score)
        random_score = []
        
        for n in range(500):
            k2 = maps2[random.randint(0,len(maps2)-1)]
            map2 = placemaps_second_curr[k2].to_numpy().flatten()
            random_score.append(np.corrcoef(map1, map2)[0][1])
        threshold = np.nanpercentile(random_score, 95)
        thresholds.append(threshold)
    
        if corr_score > threshold:
            is_placecell.append('Accepted')
        else:
            is_placecell.append('Rejected')
    
    df_placecell = pd.DataFrame([corr_scores, thresholds, is_placecell],
                                columns = maps1,
                                index = ['Score', 'P95', 'Is place cell?'])
    
    return df_placecell

def calculate_similarity(placemaps, session1, session2, savepath):
    """
    Calculate similarity scores for identified place cells between two sessions.
    
    The function reads place cell identification results from the two indicated
    sessions, and thus requires place cell identification to be done for both
    sessions prior to running it.
    

    Parameters
    ----------
    placemaps : dict
        Dictonary containing calculated placemaps for each cell.
    session1 : str
        First session to be compared.
        Format = 'sessionx', with x being an int.
    session2 : str
        Second session to be compared.
        Format = 'sessionx', with x being an int.
    savepath : path
        Directory where the place cell identification results are stored.

    Returns
    -------
    df_sim : DataFrame
        DataFrame containing similarity scores.

    """
    fname_PCID_1 = os.path.join(savepath, session1 + ' Place Cell Identification.csv')
    fname_PCID_2 = os.path.join(savepath, session2 + ' Place Cell Identification.csv')
    df_placecell_1 = pd.read_csv(fname_PCID_1, index_col = 0)
    df_placecell_2 = pd.read_csv(fname_PCID_2, index_col = 0)
    
    placecell_list = (df_placecell_1.T.loc[df_placecell_1.loc['Is place cell?']=='Accepted'].index.tolist() +
                      df_placecell_2.T.loc[df_placecell_2.loc['Is place cell?']=='Accepted'].index.tolist())
    placecell_list = list(set(placecell_list))
    placecell_list.sort()

    df_sim = pd.DataFrame(columns = placecell_list, index = ['Score'])
    
    for cell in placecell_list:
        map1 = placemaps[session1][cell].to_numpy().flatten()
        map2 = placemaps[session2][cell].to_numpy().flatten()
        if np.isnan(np.corrcoef(map1,map2)[0][1]):
            df_sim.drop(columns = cell,inplace=True)
        else:
            df_sim.loc['Score', cell] = np.corrcoef(map1, map2)[0][1]
    
    return df_sim
