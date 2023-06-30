from functions import *
from plotting import *

'''
Manual input
Modify these to match your settings
'''
data_dir = 'C:\\Users\\owner\\PlaceCell\\SampleData' # Where the data are stored
x_lens = [490, 490, 490, 490, 490] # Length of actual arena on x-axis, for each session
x_bounds = [60, 75, 60, 80, 70] # Left boundary of the actual arena, for each session
y_lens = [480,480,480,480,480] # Length of actual arena on y-axis, for each session
y_bounds = [0, 0, 0, 0, 0] # Upper boundary of the actual arena, for each session
px2cm = 490/30 # Conversion ratio between pixel and cm

# Generates subdirectory to store output
savepath = create_savepath(data_dir)

# Read data
df_traces, df_events, df_gpio, processed, n_sessions = read_inputs(
    r'C:\Users\owner\PlaceCell\SampleData')

# Extract temporal information
traces_start, traces_end, vid_start, vid_end = process_GPIO(df_gpio, n_sessions)

# Process calcium and spiking data
process_calcium(df_traces, df_events, processed, n_sessions, traces_start, traces_end)

# Process motion detection data
for s in range(n_sessions):
    df_move = movement_analysis(processed['df_vid' + str(s)],
                                processed['df_traces' + str(s)], 0.99, px2cm)
    processed['df_move' + str(s)] = df_move.copy()

# Calculate place maps for all sessions
maps_all = {}
maps_first = {}
maps_second = {}

for s in range(n_sessions):
    df_events_curr = processed['df_events' + str(s)]
    df_move_curr = processed['df_move' + str(s)]
    
    maps_all_curr = calculate_placemap(df_events_curr, df_move_curr,
                                          x_bounds[s], x_lens[s], y_bounds[s],
                                          y_lens[s], half = 'all')
    maps_all['session' + str(s)] = maps_all_curr
    
    maps_first_curr = calculate_placemap(df_events_curr, df_move_curr,
                                            x_bounds[s], x_lens[s], y_bounds[s],
                                            y_lens[s], half = 'first')
    maps_first['session' + str(s)] = maps_first_curr
    
    maps_second_curr = calculate_placemap(df_events_curr, df_move_curr,
                                             x_bounds[s], x_lens[s], y_bounds[s],
                                             y_lens[s], half = 'second')
    maps_second['session' + str(s)] = maps_second_curr
    
maps = {'all': maps_all, 'first': maps_first, 'second': maps_second}
#%%
# Place cell identification for all sessions, store results
for s in maps_first.keys():
    df_placecell = placecell_identification(maps_first[s], maps_second[s])
    df_placecell.to_csv(os.path.join(savepath, s + ' Place Cell Identification.csv'))

# Calculate similarity scores between two sessions, store results
s1 = 'session0'
s2 = 'session1'
df_sim = calculate_similarity(maps_all, s1, s2, savepath)
df_sim.to_csv(os.path.join(savepath, s1 + ' ' + s2 + ' Similarity Score.csv'))
#%%
# Plot movement trajectory for all sessions
x_res, y_res = 640, 480
plot_movement(processed, n_sessions, x_res, y_res, savepath)

# Plot distribution of speed
plot_speed_hist(processed, n_sessions, savepath)

# Plot spiking events along movement, for all sessions for indicated cells
plot_events(processed, n_sessions, x_res, y_res, savepath, cells = [' C01', ' C05'])

# Plot calcium traces of identified place cells along with speed
for s in range(n_sessions):
    session = 'session' + str(s)
    plot_traces(processed, session, savepath)

# Plot place maps for all cells, all sessions
for session in maps_all.keys():
    for cell in maps_all[session].keys():
        plot_map(maps, session, cell, savepath, half = 'all')
