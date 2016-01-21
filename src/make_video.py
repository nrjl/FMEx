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
UPDATE = 'C'

def square_cost_modifier(graph, xlo, xhi, ylo, yhi, delta):
    cost_dict={}
    for x in range(xlo, min(graph.width, xhi+1)):
        for y in range(ylo, min(graph.height, yhi+1)):
            if (x,y) not in graph.obstacles:
                cost_dict[(x,y)] = delta
    return cost_dict

def init_fig():
    fign=plt.figure()
    axn=fign.add_subplot(111)
    axn.set_aspect('equal', 'datalim')
    axn.tick_params(labelbottom='on',labeltop='off')
    axn.set_xlabel('x')
    axn.set_ylabel('y')    
    return fign, axn

print "Generating map..."
gridsize = [130, 100]

random.seed(2)

g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], fm_graphtools.blob_cost_function)
g.obstacles = fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 250, 10)
start_node = (1,1)
end_node = (127,97) #'''

# FM search
print "Performing FM search..."
figFM, axFM = init_fig()
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

## biFM search
bFM = fast_marcher.BiFastMarcher(g)
bFM.set_start(start_node)
bFM.set_goal(end_node)
bFM.set_plots([], axFM)

print "Performing biFM search..."
t0 = time.time()
bFM.search()
t_searchbFM = time.time()-t0

bFM.pull_path()
t_pathbFM = time.time()-t0-t_searchbFM

bFM.find_corridor()
t_corridorbFM = time.time()-t0-t_searchbFM-t_pathbFM

print "Done. Search took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchbFM, t_pathbFM, t_corridorbFM)


# FM search on updated map

if UPDATE == 'A':
    square_cost = square_cost_modifier(g, 50, 80, 70, 90, +1.0)
elif UPDATE == 'B':
    square_cost = square_cost_modifier(g, 30, 50, 25, 40, -1.0)
elif UPDATE == 'C':
    square_cost = square_cost_modifier(g, 40, 60, 0, 20, +1)
    square_cost.update(square_cost_modifier(g, 88, 100, 62, 90, -0.5))
elif UPDATE == 'D':
    square_cost = square_cost_modifier(g, 110, 130, 65, 80, +1.0)
    square_cost.update(square_cost_modifier(g, 0, 10, 10, 20, +1.0))
else:
    square_cost = {}

print "Performing update (FM from scratch) ..."
ug = g.copy()
ug.add_delta_costs(square_cost)
uFM = fast_marcher.FastMarcher(ug)
uFM.set_start(start_node)
uFM.set_goal(end_node)
uFM.set_plots([], axFM)

t0 = time.time()
uFM.search()
t_searchuFM = time.time()-t0

uFM.pull_path()
t_pathuFM = time.time()-t0-t_searchuFM

uFM.find_corridor()
t_corridoruFM = time.time()-t0-t_searchuFM-t_pathuFM
print "Done. Search took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchuFM, t_pathuFM, t_corridoruFM)

# E* Update test
eFM = copy.copy(FM)
eFM.set_graph(g.copy())
eFM.set_plots([], axFM)

print "Performing map update (E*) ..."
t0 = time.time()
eFM.update(square_cost, True)
t_searchFMu = time.time()-t0

eFM.pull_path()
t_pathFMu = time.time()-t0-t_searchFMu

eFM.find_corridor()
t_corridorFMu = time.time()-t0-t_searchFMu-t_pathFMu
print "Done. Update took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchFMu, t_pathFMu, t_corridorFMu)


# Update bFM
ubFM = copy.copy(bFM)
ubFM.set_graph(g.copy())
ubFM.set_plots([], axFM)

print "Performing map update (bi-FM) ..."
t0 = time.time()
ubFM.update(square_cost)
t_searchFMub = time.time()-t0

ubFM.pull_path()
t_pathFMub = time.time()-t0-t_searchFMub

ubFM.find_corridor()
t_corridorFMub = time.time()-t0-t_searchFMub-t_pathFMub
print "Done. Update took {0}s, pulling path took {1}s, extracting corridor took {2}s".format(t_searchFMub, t_pathFMub, t_corridorFMub)

