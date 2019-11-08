import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

def init_fig(graph=None, **kwargs):
    fign=plt.figure( **kwargs)
    axn=fign.add_subplot(111)
    axn.set_aspect('equal')
    axn.tick_params(labelbottom='on',labeltop='off')
    axn.set_xlabel(r'x')
    axn.set_ylabel(r'y') 
    if graph == None:
        axn.autoscale(tight=True)
    else:
        axn.set_xlim(graph.left-0.5*graph.delta[0], graph.right-.5*graph.delta[0])
        axn.set_ylim(graph.bottom-0.5*graph.delta[1], graph.top-0.5*graph.delta[1])
    return fign, axn
    
def clear_content(frame):
    for item in frame:
        item.remove()

def generate_obstacles(graph, num_obstacles, max_size=None):
    # max_size is specified in actual size in the graph, not in number of cells
    if max_size==None:
        max_size = [int(graph.nx/4), int(graph.ny/4)]
    elif isinstance(max_size, (int,float)):
        max_size = [int(max_size/graph.delta[0]), int(max_size/graph.delta[1])]
    else:
        max_size = [int(max_size[0]/graph.delta[0]), int(max_size[1]/graph.delta[1])]
    obs_list = []
    for ii in range(num_obstacles):
        x = random.randint(0, graph.nx)
        y = random.randint(0, graph.ny)
        sx = random.randint(1, max_size[0])
        sy = random.randint(1, max_size[1])
        obs_list.extend([(a, b) for a in range(x, min(graph.nx-1, x+sx)) for b in range(y, min(graph.ny, y+sy))])
    return obs_list

def make_graph_mat(graph, cost_fun, default_val=0):
    graph_mat = default_val*np.ones((graph.nx, graph.ny),dtype='float')
    for x in range(graph.nx):
        for y in range(graph.ny):
            try:
                graph_mat[x,y] = cost_fun( graph.index_to_graph((x,y)) )
            except KeyError:
                pass
    return graph_mat


def plot_end_points(axes, x, y):
    h_ends = list()
    h_ends.append(axes.plot(x[0], y[0], 'r^', markersize=8)[0])
    h_ends.append(axes.plot(x[-1], y[-1], 'ro', markersize=8 )[0])
    return h_ends


def draw_grid(axes, graph, path=None, max_cost=0, min_cost=None, cost_fun=None, *args, **kwargs):
    if cost_fun == None:
        cost_fun = graph.node_cost
    graph_mat = make_graph_mat(graph, cost_fun)
    for node in graph.obstacles:
        xi,yi = graph.graph_to_index(node)
        graph_mat[xi,yi] = -1
    graph_mat = np.ma.masked_where(graph_mat == -1, graph_mat)
    max_cost = max(max_cost, graph_mat.max())
    if min_cost == None:
        min_cost = graph_mat.min()
    cmap = plt.cm.terrain
    cmap.set_bad(color='black')
    extent = graph.get_extent()
    mat_out =  [axes.imshow(graph_mat.transpose(), origin='lower', extent=extent,
        interpolation='none', cmap=cmap, vmin=min_cost, vmax=max_cost, *args, **kwargs)]
    if not path == None:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
        mat_out.extend(plot_end_points(axes,x,y))
    axes.tick_params(labelbottom='on',labeltop='off')
    axes.set_xlim(extent[0],extent[1]); axes.set_ylim(extent[2],extent[3])
    return mat_out, [min_cost, max_cost]
    
def draw_costmap(axes, graph, cost_to_come, path=[], start_nodes=None):
    cost_mat = make_graph_mat(graph, lambda x: cost_to_come[x[0],x[1]], default_val=-1)
    for node in graph.obstacles:
        xi,yi = graph.graph_to_index(node)
        cost_mat[xi,yi] = -2
    cost_mat = np.ma.masked_where(cost_mat == -2, cost_mat)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    cmap.set_over(color='#C2A366')
    cmap.set_under(color='0.8')
    extent = graph.get_extent()
    mat_out = [axes.imshow(cost_mat.transpose(), origin='lower', extent=extent,
        norm = matplotlib.colors.Normalize(vmin=0, vmax=cost_mat.max(), clip=False))]
    if start_nodes != None:
        mat_out.append(axes.plot(start_nodes[0], start_nodes[1],'r^', markersize=8 )[0])
    if len(path) > 0:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
    axes.tick_params(labelbottom='on',labeltop='off')
    axes.set_xlim(extent[0],extent[1]); axes.set_ylim(extent[2],extent[3])
    return mat_out

def draw_corridor(axes, graph, cost_to_come, corridor, interface=[], path=[]):
    cost_mat = -1*np.ones((graph.nx, graph.ny))
    for node in corridor:
        xi,yi = graph.graph_to_index(node)
        cost_mat[xi,yi] = cost_to_come[node]
    for node in graph.obstacles:
        xi,yi = graph.graph_to_index(node)
        cost_mat[xi,yi] = -2
    max_cost = cost_mat.max()
    for node in interface:
        xi,yi = graph.graph_to_index(node)
        cost_mat[xi,yi] = max_cost+1
    cost_mat = np.ma.masked_where(cost_mat == -2, cost_mat)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    cmap.set_over(color='#C2A366')
    cmap.set_under(color='0.8')
    extent = graph.get_extent
    mat_out = [axes.imshow(cost_mat.transpose(), origin='lower',extent=extent,
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_cost, clip=False))]
        
    if len(path) > 0:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
    axes.tick_params(labelbottom='on',labeltop='off')
    axes.set_xlim(extent[0],extent[1]); axes.set_ylim(extent[2],extent[3])
    return mat_out

def draw_fbfmcost(axes, grid, path_cost, path=[], min_cost = 1e7, max_cost = 0):
    grid_mat = np.zeros((grid.width, grid.height))
    for x in range(grid.left, grid.right):
        for y in range(grid.bottom, grid.top):
            if (x,y) in path_cost:
                grid_mat[x-grid.left,y-grid.bottom] = path_cost[(x,y)]
    for x,y in grid.obstacles:
        grid_mat[x-grid.left,y-grid.bottom] = -1
    grid_mat = np.ma.masked_where(grid_mat == -1, grid_mat)
    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    axes.set_xlim([grid.left, grid.right]); axes.set_ylim([grid.bottom, grid.top])
    max_cost = max(max_cost, grid_mat.max())
    min_cost = min(min_cost, min(path_cost.values()))
    mat_out =  [axes.imshow(grid_mat.transpose(), origin='lower', extent=[grid.left,grid.right,grid.bottom,grid.top],
        interpolation='none', cmap=cmap, vmax=max_cost, vmin=min_cost)]
    if len(path) > 0:
        x, y = zip(*path)
        mat_out.append(axes.plot(x, y, 'w-', linewidth=2.0 )[0])
    axes.tick_params(labelbottom='on',labeltop='off')
    #axes.figure.colorbar(mat_out[0])
    return mat_out, [grid_mat.min(), grid_mat.max()]