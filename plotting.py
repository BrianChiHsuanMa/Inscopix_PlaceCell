import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

def plot_movement(data, n_sessions, x_res, y_res, savepath):
    """
    Plot the movement of the animal in each session.

    Parameters
    ----------
    data : dict
        The processed data.
    n_sessions : int
        Number of sessions.
    x_res : int
        Resolution on x axis, in pixels.
    y_res : int
        Resolution on y axis, in pixels.
    savepath : path
        Path where the output figures will be saved.

    Returns
    -------
    None.

    """
    for s in range(n_sessions):
        df_plot = data['df_move' + str(s)]
        ind_mid = int(df_plot.shape[0]/2)
        
        plt.figure()
        sns.lineplot(x = df_plot['head_x'], y = df_plot['head_y'], sort = False,
                     lw=1)
        plt.title('Trajectory session' + str(s))
        plt.ylim(y_res, 0)
        plt.xlim(0, x_res)
        plt.savefig(os.path.join(savepath, 'Movement trajectory session' +
                                 str(s) + '.png'))
        
        plt.figure()
        sns.lineplot(x = df_plot[:ind_mid]['head_x'],y = df_plot[:ind_mid]['head_y'],
                     sort = False, lw = 1)
        plt.title('Trajectory session' + str(s) + ' 1st')
        plt.ylim(y_res, 0)
        plt.xlim(0, x_res)
        plt.savefig(os.path.join(savepath, 'Movement trajectory session' +
                                 str(s) + ' 1st.png'))
        
        plt.figure()
        sns.lineplot(x = df_plot[ind_mid:]['head_x'], y = df_plot[ind_mid:]['head_y'],
                     sort = False, lw = 1)
        plt.title('Trajectory session' + str(s) + ' 2nd')
        plt.ylim(y_res, 0)
        plt.xlim(0, x_res)
        plt.savefig(os.path.join(savepath, 'Movement trajectory session' +
                                 str(s) + ' 2nd.png'))

def plot_speed_hist(data, n_sessions, savepath):
    """
    Plot distribution of speed in each session. Unit is cm/s.

    Parameters
    ----------
    data : dict
        The processed data.
    n_sessions : int
        Number of sessions.
    savepath : path
        Path where the output figures will be saved.

    Returns
    -------
    None.

    """
    for s in range(n_sessions):
        df_move = data['df_move' + str(s)]
        plt.figure()
        sns.displot(data = df_move, x = 'Speed', kde = True, stat 
                    = 'probability', binwidth = 0.5, height = 5, aspect = 2)
        plt.title('Speed distribution session' + str(s))
        plt.savefig(os.path.join(savepath, 'Speed distribution session' +
                                 str(s) + '.png'), bbox_inches='tight')

def plot_events(data, n_sessions, x_res, y_res, savepath, cells = []):
    """
    

    Parameters
    ----------
    data : dict
        The processed data.
    n_sessions : int
        Number of sessions.
    x_res : int
        Resolution on x axis, in pixels.
    y_res : int
        Resolution on y axis, in pixels.
    savepath : path
        Path where the output figures will be saved.
    cells : list, optional
        List of cells whose spiking events will be plotted.
        The default is [], which will plot all the cells.

    Returns
    -------
    None.

    """
    for s in range(n_sessions):
        df_events = data['df_events' + str(s)]
        df_move = data['df_move' + str(s)]
        
        if not cells:
            cells = df_events.columns[1:].to_list()
            
        for cell in cells:
            plt.figure()
            ax = plt.subplot(111)
            ax.axis('off')
            ind_spk = df_events.loc[df_events[cell] > 0].index
            ind_run = df_move.loc[df_move['Speed'] >= 0.5].index
            ind_plot = ind_spk.intersection(ind_run)
    
            sns.lineplot(x = df_move['head_x'], y = df_move['head_y'],
                         sort = False, lw = 1)
            plt.plot(df_move.loc[ind_plot, 'head_x'], df_move.loc[ind_plot, 'head_y'],
                     'ro', markersize=5)
            plt.title('Events session' + str(s) + cell)
            plt.ylim(y_res, 0)
            plt.xlim(0, x_res)
            plt.tight_layout()
            plt.savefig(os.path.join(savepath, cell + ' Events session' +
                                     str(s) + '.png'))

def plot_traces(data, session, savepath, place_cell_only = True, cell_list = []):
    """
    Plot calcium traces and speed with time.

    Parameters
    ----------
    data : dict
        The processed data.
    session : str
        The session to be plotted.
        Format = 'sessionx', with x being an int.
    savepath : path
        Path where the output figures will be saved.
    place_cell_only : bool, optional
        Whether to plot the place cells only.
        The default is True, which will only plot the identified place cells
        of that session. This will also override cell_list
    cell_list : list, optional
        List of cells to plot. The default is [].

    Returns
    -------
    None.

    """
    if place_cell_only:
        fname_PCID = os.path.join(savepath, session + ' Place Cell Identification.csv')
        df_placecell = pd.read_csv(fname_PCID, index_col = 0)
        cell_list = (df_placecell.T.loc[df_placecell.loc['Is place cell?']=='Accepted'].index.tolist())
        cell_list.sort()
        
    s = session[-1]
    df_traces = data['df_traces' + s]
    df_move = data['df_move' + s]
    
    scaler = MinMaxScaler()
    df_traces_scale = pd.DataFrame(scaler.fit_transform(df_traces.iloc[:, 1:]),
                                 columns = df_traces.columns[1:],
                                 index = df_traces.index)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    
    for n, cell in enumerate(cell_list):
        ax.plot(df_traces['Time (s)'], df_traces_scale[cell] + n, label = cell)
    ax.plot(df_traces['Time (s)'], df_move['Speed'] / np.max(df_move['Speed'])
            + n + 3, 'k', label = 'Speed')
    
    plt.xlabel('Time(s)')
    plt.title('Traces ' + session)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc = 'upper right',
              bbox_to_anchor = (1.25, 1.05), fancybox = True, shadow = True)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    
    plt.savefig(os.path.join(savepath, 'Traces ' + session + '.png'))
    
def plot_map(maps, session, cell, savepath, half = 'all'):
    """
    Plot place map of a cell.

    Parameters
    ----------
    maps : dict
        Dictionary containing all the place maps.
    session : str
        The session to be plotted.
        Format = 'sessionx', with x being an int.
    cell : str
        Name of the cell to be plotted.
    savepath : path
        Path where the output figures will be saved.
    half : str, optional
        Whether to plot the place map of half a session. The default is 'all'.
        Accepted values:
            'all': The entire session.
            'first': The first half.
            'second': The second half.

    Returns
    -------
    None.

    """
    plt.figure()
    ax = plt.subplot(111)
    df_plot = maps[half][session][cell]
    sns.heatmap(df_plot, square=True, cmap='plasma', vmin=0, vmax=1)
    plt.title('Place map ' + session + cell + ' ' + half)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, cell + ' Place map' + ' ' + session
                             + '_' + half + '.png'))