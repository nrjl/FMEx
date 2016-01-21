import matplotlib.pyplot as plt
import numpy as np
import random

import sys
sys.path.append('../src')
import fm_graphtools
import fm_plottools

def square_cost_modifier(graph, xlo, xhi, ylo, yhi, delta):
    cost_dict={}
    for x in range(xlo, min(graph.width, xhi+1)):
        for y in range(ylo, min(graph.height, yhi+1)):
            if (x,y) not in graph.obstacles:
                cost_dict[(x,y)] = delta
    return cost_dict
    
FIG_DIR = '/home/nick/Dropbox/work/FastMarching/fig/'

gridsize = [130, 100]
random.seed(2)
g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], fm_graphtools.blob_cost_function)

g.obstacles = fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 250, 10)
start_node = (1,1)
end_node = (127,97) 

fign=plt.figure()
axn=fign.add_subplot(111)
axn.set_aspect('equal', 'datalim')
axn.tick_params(labelbottom='on',labeltop='off')
axn.set_xlabel('x')
axn.set_ylabel('y')    


def draw_grid(axes, grid, bmin, bmax):
    grid_mat = np.zeros((grid.width, grid.height))
    for x in range(grid.width):
        for y in range(grid.height):
            grid_mat[x,y] = grid.node_cost((x,y))
    for x,y in grid.obstacles:
        grid_mat[x,y] = -1
    grid_mat = np.ma.masked_where(grid_mat == -1, grid_mat)
    cmap = plt.cm.terrain
    cmap.set_bad(color='black')
    axes.set_xlim([0, grid.width]); axes.set_ylim([0, grid.height])
    mat_out =  [axes.matshow(grid_mat.transpose(), interpolation='none', cmap=cmap, vmin=bmin, vmax=bmax)]
    axes.tick_params(labelbottom='on',labeltop='off')
    return mat_out, [grid_mat.min(), grid_mat.max()]

def get_update(UPDATE):
    if UPDATE == 'A':
        square_cost = square_cost_modifier(g, 50, 80, 70, 90, +1.0)
        bb = [[50, 80, 70, 90, 1]]
    elif UPDATE == 'B':
        square_cost = square_cost_modifier(g, 30, 50, 25, 40, -1.0)
        bb = [[30, 50, 25, 40, -1]]
    elif UPDATE == 'C':
        square_cost = square_cost_modifier(g, 40, 60, 0, 20, +1)
        square_cost.update(square_cost_modifier(g, 88, 100, 62, 90, -0.5))
        bb = [[40, 60, 0, 20, 1], [88, 100, 62, 90, -1]]
    elif UPDATE == 'D':
        square_cost = square_cost_modifier(g, 110, 130, 65, 80, +1.0)
        square_cost.update(square_cost_modifier(g, 0, 10, 10, 20, +1.0))
        bb = [[110, 130, 65, 80, 1], [0, 10, 10, 20, 1]]
    else:
        square_cost = {}
    return square_cost,bb

fign.set_size_inches(9,6)
graph_frame, barlims = draw_grid(axn, g, 0.5, 3.5)
cbar = fign.colorbar(graph_frame[0], ticks=[0.5 + x*0.5 for x in range(8)])
axn.plot([start_node[0]], [start_node[1]], '^r',markersize=8)
axn.plot([end_node[0]], [end_node[1]], 'or',markersize=8)
fign.savefig(FIG_DIR+'graph_cost.pdf', bbox_inches='tight')    
fign.clf()
axn=fign.add_subplot(111)
axn.set_aspect('equal', 'datalim')
axn.tick_params(labelbottom='on',labeltop='off')
axn.set_xlabel('x')
axn.set_ylabel('y')
fign.set_size_inches(5,4)

for UPDATE in ['A', 'B', 'C', 'D']:
    ug = g.copy()
    cu,bb  = get_update(UPDATE)
    ug.add_delta_costs(cu)
    axn.clear()
    graph_frame, barlims = draw_grid(axn, ug, 0.5, 3.5)
    for box in bb:
        if box[4] > 0: col = 'r'
        else: col = 'w'
        axn.plot([box[0], box[1], box[1], box[0], box[0]], [box[2], box[2], box[3], box[3], box[2]], '--'+col)
    fign.savefig(FIG_DIR+UPDATE+'graph_cost.pdf', bbox_inches='tight')    