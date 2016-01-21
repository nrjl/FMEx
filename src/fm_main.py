import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
import copy
import fast_marcher

import fm_graphtools
import fm_plottools
# from dijkstra_search import dijkstra_search, pull_path

VID_DIR = '/home/nick/Dropbox/work/FastMarching/vid/'
FIG_DIR = '/home/nick/Dropbox/work/FastMarching/fig2/'
UPDATE = 'D'
CREATE_FIGS = True

print "Generating map..."
gridsize = [130, 100]

random.seed(2)

g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], fm_graphtools.blob_cost_function)

'''g.obstacles = generate_obstacles(gridsize[0], gridsize[1], 25, 25)
start_node = (1,1)
end_node = (110, 90) #'''

g.obstacles = fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 250, 10)
start_node = (1,1)
end_node = (127,97) #'''

'''g.obstacles = zip(70*np.ones((40,), np.int), range(31, 71))
start_node = (1, 50)
end_node = (100, 50) #'''

# FM search
print "Performing FM search..."
figFM, axFM = fm_plottools.init_fig()
FM = fast_marcher.FastMarcher(g)
FM.set_start(start_node)
FM.set_goal(end_node)
FM.set_plots([], axFM)

t0 = time.time()
FM.search()
t_searchFM = time.time()-t0

FM.pull_path()
t_pathFM = time.time()-t0-t_searchFM

FM.find_corridor()
t_corridorFM = time.time()-t0-t_searchFM-t_pathFM
print "Done. Search took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchFM, t_pathFM, t_corridorFM)

if CREATE_FIGS:
    FM.make_pictures(FIG_DIR+UPDATE)

aniFM = animation.ArtistAnimation(figFM, FM.make_video(), interval=10, repeat_delay=2000)
# aniFM.save(VID_DIR+'fast_march.mp4', writer = 'avconv', fps=50, bitrate=1500)

## biFM search
figbFM, axbFM = fm_plottools.init_fig()
bFM = fast_marcher.BiFastMarcher(g)
bFM.set_start(start_node)
bFM.set_goal(end_node)
bFM.set_plots([], axbFM)

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
    bFM.make_pictures(FIG_DIR+UPDATE+'b')

anibFM = animation.ArtistAnimation(figbFM, bFM.make_video(), interval=10, repeat_delay=2000)
# anibFM.save(VID_DIR+'bi_fast_march.mp4', writer = 'avconv', fps=50, bitrate=1500)

# FM search on updated map

if UPDATE == 'A':
    square_cost = fm_graphtools.square_cost_modifier(g, 50, 80, 70, 90, +1.0)
elif UPDATE == 'B':
    square_cost = fm_graphtools.square_cost_modifier(g, 30, 50, 25, 40, -1.0)
elif UPDATE == 'C':
    square_cost = fm_graphtools.square_cost_modifier(g, 40, 60, 0, 20, +1)
    square_cost.update(fm_graphtools.square_cost_modifier(g, 88, 100, 62, 90, -0.5))
elif UPDATE == 'D':
    square_cost = fm_graphtools.square_cost_modifier(g, 110, 130, 65, 80, +1.0)
    square_cost.update(fm_graphtools.square_cost_modifier(g, 0, 10, 10, 20, +1.0))
else:
    square_cost = {}

print "Performing update (FM from scratch) ..."
figuFM, axuFM = fm_plottools.init_fig()
ug = g.copy()
ug.add_delta_costs(square_cost)
uFM = fast_marcher.FastMarcher(ug)
uFM.set_start(start_node)
uFM.set_goal(end_node)
uFM.set_plots([], axuFM)

t0 = time.time()
uFM.search()
t_searchuFM = time.time()-t0

uFM.pull_path()
t_pathuFM = time.time()-t0-t_searchuFM

uFM.find_corridor()
t_corridoruFM = time.time()-t0-t_searchuFM-t_pathuFM
print "Done. Search took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchuFM, t_pathuFM, t_corridoruFM)

if CREATE_FIGS:
    uFM.make_pictures(FIG_DIR+UPDATE+'u')
    
path_frameuFM, TEMP = fm_plottools.draw_grid(uFM.axes, FM.graph, FM.path)
aniuFM = animation.ArtistAnimation(figuFM, uFM.make_video(path_frameuFM), interval=10, repeat_delay=2000)

# E* Update test
figeFM, axeFM = fm_plottools.init_fig()
eFM = copy.copy(FM)
eFM.set_graph(g.copy())
eFM.set_plots([], axeFM)

print "Performing map update (E*) ..."
t0 = time.time()
eFM.update(square_cost, True)
t_searchFMu = time.time()-t0

eFM.pull_path()
t_pathFMu = time.time()-t0-t_searchFMu

eFM.find_corridor()
t_corridorFMu = time.time()-t0-t_searchFMu-t_pathFMu
print "Done. Update took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchFMu, t_pathFMu, t_corridorFMu)

if CREATE_FIGS:
    eFM.make_pictures(FIG_DIR+UPDATE+'e')
    
path_frameFM, TEMP = fm_plottools.draw_grid(eFM.axes, FM.graph, FM.path)
anieFM = animation.ArtistAnimation(figeFM, eFM.make_video(path_frameFM), interval=10, repeat_delay=2000)
# anieFM.save(VID_DIR+'e_star_update.mp4', writer = 'avconv', fps=50, bitrate=1500)


# Update bFM
figubFM, axubFM = fm_plottools.init_fig()
ubFM = copy.copy(bFM)
ubFM.set_graph(g.copy())
ubFM.set_plots([], axubFM)

print "Performing map update (bi-FM) ..."
t0 = time.time()
ubFM.update(square_cost)
t_searchFMub = time.time()-t0

ubFM.pull_path()
t_pathFMub = time.time()-t0-t_searchFMub

ubFM.find_corridor()
t_corridorFMub = time.time()-t0-t_searchFMub-t_pathFMub
print "Done. Update took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchFMub, t_pathFMub, t_corridorFMub)

if CREATE_FIGS:
    ubFM.make_pictures(FIG_DIR+UPDATE+'ub')

path_frameFM, TEMP = fm_plottools.draw_grid(ubFM.axes, bFM.graph, bFM.path)
aniubFM = animation.ArtistAnimation(figubFM, ubFM.make_video(path_frameFM), interval=10, repeat_delay=2000)
# aniubFM.save(VID_DIR+'bidirectional_fast_march_update.mp4', writer = 'avconv', fps=50, bitrate=1500)



plt.show()