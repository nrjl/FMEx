import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import time
import math
import copy

import fast_marcher
import fm_graphtools
import fm_plottools

# from dijkstra_search import dijkstra_search, pull_path

VID_DIR = '/home/nick/Dropbox/work/FastMarching/vid/'
FIG_DIR = '/home/nick/Dropbox/work/FastMarching/fig_explore/'
CREATE_FIGS = False

def explore_cost_function(a, b):
    cost = 3 
    cost += 10*math.exp(-math.sqrt((a-40)**2 + (b-40)**2)/16)
    cost += 5*math.exp(-math.sqrt((a-20)**2 + (b-80)**2)/8)
    cost += 4*math.exp(-math.sqrt((a-80)**2 + (b-60)**2)/32)
    cost += 7*math.exp(-math.sqrt((a+20)**2 + (b-50)**2)/32)
    cost += 7*math.exp(-math.sqrt((a-120)**2 + (b-50)**2)/32)
    return cost
    
print "Generating map..."
gridsize = [100, 100]

random.seed(6)

g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], explore_cost_function)

g.obstacles = []; #fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 30, 25)
start_node = (3,3)
end_node = (97, 97)

## biFM search
# figbFM, axbFM = init_fig()
bFM = fast_marcher.BiFastMarcher(g)
bFM.set_start(start_node)
bFM.set_goal(end_node)
# bFM.set_plots([], axbFM)

print "Performing biFM search..."
t0 = time.time()
bFM.search()
t_searchbFM = time.time()-t0

bFM.pull_path()
t_pathbFM = time.time()-t0-t_searchbFM

bFM.find_corridor()
t_corridorbFM = time.time()-t0-t_searchbFM-t_pathbFM

print "Done. Search took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchbFM, t_pathbFM, t_corridorbFM)

if CREATE_FIGS:
    bFM.make_pictures(FIG_DIR)

# anibFM = animation.ArtistAnimation(figbFM, bFM.make_video(), interval=10, repeat_delay=2000)
# anibFM.save(VID_DIR+'bi_fast_march.mp4', writer = 'avconv', fps=50, bitrate=1500)

fbFM = fast_marcher.FullBiFastMarcher(g)
fbFM.set_start(start_node)
fbFM.set_goal(end_node)

print "Performing Full biFM search..."
t0 = time.time()
fbFM.search()
t_searchfbFM = time.time()-t0

fbFM.pull_path()
t_pathfbFM = time.time()-t0-t_searchfbFM

#fbFM.find_corridor()
#t_corridorfbFM = time.time()-t0-t_searchfbFM-t_pathfbFM

figbFM, axbFM = fm_plottools.init_fig()
figfbFM, axfbFM = fm_plottools.init_fig()
fm_plottools.draw_costmap(axbFM, g, bFM.cost_to_come, bFM.path)
fm_plottools.draw_fbfmcost(axfbFM, g, fbFM.path_cost, fbFM.path)

print "Done. Search took {0}s, pulling path took {1}s".format(t_searchfbFM, t_pathfbFM)


## UPDATES!
#cost_update = fm_graphtools.square_cost_modifier(g, 60, 80, 10, 30, -3)


test_gridx = range(10, 100, 10); lx = len(test_gridx)
test_gridy = range(10, 100, 10); ly = len(test_gridy)
delta_costs = [-2, 2]; ld = len(delta_costs)

NUM_TESTS = lx*ly*ld

search_time = np.zeros([2, NUM_TESTS], float)
search_nodes = np.zeros([2, NUM_TESTS], int)
downwind_nodes = np.zeros([2, NUM_TESTS], int)

ubFM = copy.copy(bFM)
ubFM.set_graph(g.copy())

for ii in range(NUM_TESTS):
    td = delta_costs[int (ii/(lx*ly))]
    tx = test_gridx[int ((ii%(lx*ly))/len(test_gridx))]
    ty = test_gridy[(ii%(lx*ly)) % len(test_gridy)]
    #print "Polynomial update at ({0}, {1}), delta = {2}".format(tx, ty, td)
    cost_update = fm_graphtools.polynomial_cost_modifier(g, tx, ty, 20, td)
    
    #fast_marcher.bFM_reset(ubFM, bFM, g)
    #t0 = time.time()
    #ubFM.update(cost_update)
    #search_time[0,ii] = time.time()-t0
    #search_nodes[0,ii] = ubFM.search_nodes
    #downwind_nodes[0,ii] = ubFM.downwind_nodes
    #print " BiFM  |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[0,ii], search_nodes[0,ii],downwind_nodes[0,ii],ubFM.best_cost)
    
    t0 = time.time()
    fbFM.update(cost_update)
    search_time[1,ii] = time.time()-t0
    search_nodes[1,ii] = fbFM.search_nodes
    #print " FBFM  |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[1,ii], search_nodes[1,ii],downwind_nodes[1,ii],fbFM.updated_min_path_cost)

ubFM.pull_path()
figubFM, axubFM = fm_plottools.init_fig()
figufbFM, axufbFM = fm_plottools.init_fig()
fm_plottools.draw_grid(axubFM, g, ubFM.path)
fm_plottools.draw_grid(axufbFM, g, fbFM.updated_path)

plt.show()