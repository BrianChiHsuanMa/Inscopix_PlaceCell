# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:38:29 2023

@author: owner
"""
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

from functions import *

savepath = create_savepath(r'C:\Users\owner\PlaceCell\SampleData')
df_traces, df_events, df_gpio, processed, n_sessions = read_inputs(
    r'C:\Users\owner\PlaceCell\SampleData')
traces_start, traces_end, vid_start, vid_end = process_GPIO(df_gpio, n_sessions)
process_calcium(df_traces, df_events, processed, n_sessions, traces_start, traces_end)
for s in range(n_sessions):
    df_move = movement_analysis(processed['df_vid' + str(s)],
                                processed['df_traces' + str(s)])
    processed['df_move' + str(s)] = df_move.copy()

x_lens = [490, 490, 490, 490, 490]
x_bounds = [60, 75, 60, 80, 70]
y_lens = [480,480,480,480,480]
y_bounds = [0, 0, 0, 0, 0]
placemaps_all = {}
placemaps_first = {}
placemaps_second = {}
for s in range(n_sessions):
    df_events_curr = processed['df_events' + str(s)]
    df_move_curr = processed['df_move' + str(s)]
    placemaps_all_curr = calculate_placemap(df_events_curr, df_move_curr,
                                          x_bounds[s], x_lens[s], y_bounds[s],
                                          y_lens[s], half = 'all')
    placemaps_all['session' + str(s)] = placemaps_all_curr
    placemaps_first_curr = calculate_placemap(df_events_curr, df_move_curr,
                                            x_bounds[s], x_lens[s], y_bounds[s],
                                            y_lens[s], half = 'first')
    placemaps_first['session' + str(s)] = placemaps_first_curr
    placemaps_second_curr = calculate_placemap(df_events_curr, df_move_curr,
                                             x_bounds[s], x_lens[s], y_bounds[s],
                                             y_lens[s], half = 'second')
    placemaps_second['session' + str(s)] = placemaps_second_curr
    
for s in placemaps_first.keys():
    df_placecell = placecell_identification(placemaps_first[s], placemaps_second[s])
    df_placecell.to_csv(os.path.join(savepath, s + ' Place Cell Identification.csv'))