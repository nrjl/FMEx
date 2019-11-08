import numpy as np
import random
import time
import math
import bfm_explorer
from fm_tools import fast_marcher, fm_graphtools, fm_plottools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)
plt.close('all')

gridsize = [31,46]    # Grid size
SEED_NUM = 6
random.seed(SEED_NUM)           # Random seed
np.random.seed(SEED_NUM)
PRE_SAMPLES = 10
fig_size = [2.5,4]

OBSTACLES_ON = False
SEP_PLOTS = True
SAVE_PLOTS = False

mean_value = 5.0           # Mean value of the field for GP
num_blobs = 20          # Number of blobs in each field
peak_range = [-3.0,8.0]      # Low and high peak values for map blobs
spread_range = [5,12]    # Low and high spread distances for map blobs
poly_cost_radius = 20

OBS_STD = 0.05
GP_l=6.0; GP_sv=15.0; GP_sn=0.1

num_obstacles = 10       # Total number of obstacles
obstacle_size = 10       # Obstacle size

newX = [10,35]

triple_path_blobs = [[35,15,20,-2],[80,30,17,-2],[60,35,12,5],[40,55,8,6],
                     [20,45,8,-1.0],[75,75,5,5],[85,70,5,-2],[55,75,6,-1.5],
                     [10,90,6,-1]]

def explore_cost_function(a, b, blobs):
    # blobs are defined by [xloc, yloc, spread, peakval]
    # blobs = triple_path_blobs
    cost = mean_value
    for i in range(np.shape(blobs)[0]):
        cost += blobs[i][3]*math.exp(-math.sqrt((a-blobs[i][0])**2 + (b-blobs[i][1])**2)/blobs[i][2])
    return cost
    
def sample_cost_fun(cf, x, blobs):
    y = cf(x[0], x[1], blobs) + random.normalvariate(0, OBS_STD)
    return max(0, y)
    
def sample_value(fm, pc, x, y):
    std = math.sqrt(fm.GP_cost_graph.var_fun(x,y))
    ccu = fm.cost_update_new(pc.set_update(x, y, std))
    ccd = fm.cost_update_new(pc.set_update(x, y, -std))
    print "At ({0:2n},{1:2n}), std={2:4.2f}, c_up = {3:4.2f}, c_down = {4:4.2f}, c = {5:4.2f}".format(x,y,std,ccu,ccd,ccu+ccd)
    return ccu+ccd

def draw_circle(ax,c,r):
    ntheta = 180
    theta = np.linspace(0,2*np.pi,ntheta)
    xx = c[0] + r*np.cos(theta)
    yy = c[1] + r*np.sin(theta)
    ax.plot(xx,yy,'k--',dashes=(3,1))

true_g = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], cost_fun=explore_cost_function, obstacles=[])
if OBSTACLES_ON:
    true_g.update_obstacles(fm_plottools.generate_obstacles(true_g, num_obstacles, obstacle_size))
start_node = (1,1)
while start_node in true_g.obstacles:
    start_node = (start_node[0]+1, start_node[1])
end_node = (gridsize[0]-3, gridsize[1]-3)
while end_node in true_g.obstacles:
    end_node = (end_node[0]-1, end_node[1])

cblobs = []
for ii in range(num_blobs):
    cblobs.append([random.uniform(-10,gridsize[0]+10), random.uniform(-10,gridsize[1]+10), 
        random.uniform(spread_range[0], spread_range[1]), random.uniform(peak_range[0], peak_range[1])])
explorer_cost = bfm_explorer.mat_cost_function(true_g, explore_cost_function, cblobs)
true_g.cost_fun = explorer_cost.calc_cost
tFM = fast_marcher.FastMarcher(true_g)
tFM.set_start(start_node)
tFM.set_goal(end_node)
tFM.search()
tFM.pull_path()
    
poly_cost_obj = fm_graphtools.polynomial_precompute_cost_modifier(true_g, poly_cost_radius, min_val=0.001)

X = np.random.random((PRE_SAMPLES, 2))*(true_g.width-1)
#X = np.vstack([X,[44,38],[89,73],[59,54],[70,71],[73,38],[52,28],[41,3],[84,94],[30,55],[80,20]]) # [20,26],[37,44],[84,72],[90,90]  [30,30],[50,50],[60,60],[70,70],
Y = np.atleast_2d([sample_cost_fun(explore_cost_function, x, cblobs) for x in X]).transpose()

FMEx = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value=mean_value, obs=true_g.obstacles,
    GP_l=GP_l, GP_sv=GP_sv, GP_sn=GP_sn)
FMEx.search()

if SEP_PLOTS:
    f0 = [plt.figure(i, figsize=fig_size) for i in range(4)]
    a0 = [f0[i].add_subplot(111) for i in range(4)]
else:
    f0 = plt.figure(0); f0.clf()
    a0 = [f0.add_subplot(2,2,i+1) for i in range(4)]

temp,barlims=fm_plottools.draw_grid(a0[0], true_g, path=tFM.path)
fm_plottools.draw_grid(a0[1], FMEx.GP_cost_graph, min_cost=barlims[0], max_cost=barlims[1])
a0[1].plot(X[:,0], X[:,1], 'rx')

newY = explore_cost_function(newX[0], newX[1], cblobs)
delta = newY - FMEx.GP_cost_graph.cost_fun(newX[0], newX[1])

cost_update = poly_cost_obj.set_update(newX[0], newX[1], delta)
ts = time.time()
FMEx.cost_update_new(cost_update, recalc_path=True)
tup = time.time()-ts
fm_plottools.draw_grid(a0[2], FMEx.fbFM.graph, min_cost=barlims[0], max_cost=barlims[1])
a0[2].plot(newX[0], newX[1], 'ko')
draw_circle(a0[2], newX, poly_cost_radius*.5)
a0[2].plot(X[:,0], X[:,1], 'rx')
ts = time.time()
FMEx.add_observation(np.array([newX]), np.array([[newY]]))
tgp = time.time()-ts
fm_plottools.draw_grid(a0[3], FMEx.GP_cost_graph,min_cost=barlims[0], max_cost=barlims[1])
a0[3].plot(FMEx.X[:,0], FMEx.X[:,1], 'rx')
a0[3].plot(newX[0], newX[1], 'ko', fillstyle='none')

for aa in a0:
    aa.set_xticks(range(0,gridsize[0]+1,10))

print "Poly update took {0}s, full GP update took {1}s".format(tup,tgp)

if SEP_PLOTS and SAVE_PLOTS:
    for i in range(len(f0)):
        f0[i].savefig('../fig/poly_update{0}.pdf'.format(i), bbox_inches='tight')

#fm_plottools.draw_grid(a1[1], FMEx.GP_cost_graph, min_cost=barlims[0], max_cost=barlims[1])
plt.show()
