import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fm_graphtools
import fm_plottools
import bfm_explorer
import fast_marcher
import random
import math
import time

GP_l=8.0; GP_sv=15.0; GP_sn=0.1

#from explorer_statrun import explore_cost_function, sample_cost_fun, calc_true_path_cost, calc_est_path_cost
def explore_cost_function(a, b, blobs):
    # blobs are defined by [xloc, yloc, spread, peakval]
    # blobs = my_blobs
    cost = mean_value
    for i in range(np.shape(blobs)[0]):
        cost += blobs[i][3]*math.exp(-math.sqrt((a-blobs[i][0])**2 + (b-blobs[i][1])**2)/blobs[i][2])
    return cost
    
def sample_cost_fun(cf, x, blobs):
    y = cf(x[0], x[1], blobs) + random.normalvariate(0, 0.1)
    return max(0, y)

def calc_true_path_cost(cost_fun, path, *args):
    true_cost = 0
    for i in range(len(path)-1):
        d = np.sqrt((path[i][0]-path[i+1][0])**2 + (path[i][1]-path[i+1][1])**2)
        true_cost += (cost_fun(path[i][0], path[i][1], *args) + 
            cost_fun(path[i+1][0], path[i+1][1], *args))/2*d
    return true_cost

def calc_est_path_cost(gp_model, mean_val, path):
    true_cost,var_cost = gp_model.predict(np.asarray(path))
    true_cost += mean_val
    sum_cost = 0
    sum_var = 0
    for i in range(len(path)-1):
        d = np.sqrt((path[i][0]-path[i+1][0])**2 + (path[i][1]-path[i+1][1])**2)
        sum_cost += (true_cost[i] + true_cost[i+1])/2*d
        sum_var += (var_cost[i] + var_cost[i+1])/2*d  
    return sum_cost, sum_var

def sample_value_new(fm, pc, x, y, delta_costs):
    std = math.sqrt(fm.GP_cost_graph.var_fun(x,y))
    cc = 0
    for dc in delta_costs:
        cc += fm.cost_update_new(pc.set_update(x, y, dc*std),loc=[x,y])
    return cc
    
MAKE_VIDEO = True
MAKE_PICTURES=True
gridsize = [100, 100]
mean_value = 5.0
NUM_SAMPLES = 1
OBSTACLES_ON = True
num_obstacles = 40
obstacle_size = 10
delta_costs = [-1, 0.5]
random.seed(2)

num_blobs = 30           # Number of blobs in each field
peak_range = [-3.0,8.0]      # Low and high peak values for map blobs
spread_range = [5,12]    # Low and high spread distances for map blobs

VID_DIR = '../vid/'

true_g = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], cost_fun=explore_cost_function, obstacles=[])
if OBSTACLES_ON:
    true_g.update_obstacles(fm_plottools.generate_obstacles(true_g, num_obstacles, obstacle_size))
start_node = (3,3)
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

poly_cost_obj = fm_graphtools.polynomial_precompute_cost_modifier(true_g, 13, min_val=0.001)
        
start_node = (3,3)
end_node = (97, 97)
while start_node in true_g.obstacles:
    start_node = (start_node[0]+1, start_node[1])
while end_node in true_g.obstacles:
    end_node = (end_node[0]-1, end_node[1])

X = np.array([[3,3],[80,95], [55,45], [25,30], [38,60], [52,30],[65,70],[37,45],[14,41],[80,30],[83,85],[97,63]])
Xshape = X.shape
Y = np.zeros((Xshape[0], 1))
for ii in range(Xshape[0]):
    Y[ii] = sample_cost_fun(explore_cost_function, X[ii,:], cblobs)

fm_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value=mean_value, obs=true_g.obstacles,
    GP_l=GP_l, GP_sv=GP_sv, GP_sn=GP_sn)

if MAKE_VIDEO:
    t_frames = []
    fig_sg, ax_sg = fm_plottools.init_fig(graph=true_g, figsize=(6,5))
    fm_sampling_explorer.set_plots(t_frames, ax_sg)
    fm_sampling_explorer.set_plot_costs(0, 15)
fm_sampling_explorer.search()

if MAKE_PICTURES:
    tFM = fast_marcher.FullBiFastMarcher(true_g)
    tFM.set_start(start_node)
    tFM.set_goal(end_node)
    tFM.search()
    tFM.pull_path()
    fig1,ax1 = fm_plottools.init_fig(graph=true_g, figsize=(6,5))
    tempframe,barlims  = fm_plottools.draw_grid(ax1, true_g, tFM.path,min_cost=mean_value-0.5)
    cbar = fig1.colorbar(tempframe[0], shrink=0.7)
    fig1.savefig(VID_DIR+'true_map.png', bbox_inches='tight', transparent=True)
    fm_plottools.clear_content(tempframe)
    tempframe,barlims= fm_plottools.draw_grid(ax1, fm_sampling_explorer.GP_cost_graph, min_cost=barlims[0], max_cost=barlims[1])
    tempframe.append(ax1.plot(fm_sampling_explorer.X[:,0], fm_sampling_explorer.X[:,1], 'rx', mew=1)[0])
    #cbar = fig1.colorbar(tempframe[0], shrink=0.7)
    fig1.savefig(VID_DIR+'initial_estimate.png', bbox_inches='tight', transparent=True)
    tempframe.append(ax1.plot(45, 16, 'wo', mfc='none', ms=12, mew=2, mec='w')[0])
    fig1.savefig(VID_DIR+'update_location.png', bbox_inches='tight', transparent=True)
    fm_plottools.clear_content(tempframe)
    tempframe,barlims= fm_plottools.draw_grid(ax1, fm_sampling_explorer.GP_cost_graph, fm_sampling_explorer.fbFM.path, min_cost=barlims[0], max_cost=barlims[1])
    tempframe.append(ax1.plot(fm_sampling_explorer.X[:,0], fm_sampling_explorer.X[:,1], 'rx', mew=1)[0])
    fig1.savefig(VID_DIR+'initial_estimate_withpath.png', bbox_inches='tight', transparent=True)
    
true_path_cost = np.zeros(NUM_SAMPLES)
est_path_cost = np.zeros(NUM_SAMPLES)
est_path_var = np.zeros(NUM_SAMPLES)

for ii in range(NUM_SAMPLES):
    
    fm_best_cost = -1   
    ts = time.time()
    
    rand_samples = []
    while len(rand_samples) < 50:
        tx = random.choice(range(gridsize[0]))
        ty = random.choice(range(gridsize[1]))
        if ((tx,ty) not in true_g.obstacles):
            rand_samples.append((tx,ty))
        
    ts = time.time()
    sv = [sample_value_new(fm_sampling_explorer, poly_cost_obj, x[0],x[1], delta_costs) for x in rand_samples]
    #print "Update new sample selection time: {0}s".format(time.time()-ts)
    imin = np.argmin(sv)
    fm_best_cost = sv[imin]
    fm_bestX = rand_samples[imin]        
    print "Sample selection time: {0:.4f}".format(time.time()-ts)                                        

    fm_sampling_explorer.add_observation([fm_bestX], [[sample_cost_fun(explore_cost_function, fm_bestX, cblobs)]])
    true_path_cost[ii] = calc_true_path_cost(explore_cost_function, fm_sampling_explorer.fbFM.path, cblobs)
    est_path_cost[ii],est_path_var[ii] = calc_est_path_cost(fm_sampling_explorer.GP_model, mean_value, fm_sampling_explorer.fbFM.path)

if MAKE_VIDEO:
    ani = animation.ArtistAnimation(fig_sg, fm_sampling_explorer.fbFM.image_frames, interval=20, repeat_delay=2000)
    # ani.save(VID_DIR+'search_video.mp4', writer = 'avconv', fps=25, bitrate=5000)

if MAKE_PICTURES:
    fm_plottools.clear_content(tempframe)
    tempframe,barlims= fm_plottools.draw_grid(ax1, fm_sampling_explorer.GP_cost_graph, min_cost=barlims[0], max_cost=barlims[1])
    tempframe.append(ax1.plot(fm_sampling_explorer.X[:,0], fm_sampling_explorer.X[:,1], 'rx', mew=1)[0])
    tempframe.append(ax1.plot(fm_bestX[0], fm_bestX[1], 'wo', mfc='none', ms=12, mew=1, mec='w')[0])
    #cbar = fig1.colorbar(tempframe[0], shrink=0.7)
    fig1.savefig(VID_DIR+'new_estimate.png', bbox_inches='tight', transparent=True)
    fm_plottools.clear_content(tempframe)
    tempframe,barlims= fm_plottools.draw_grid(ax1, fm_sampling_explorer.GP_cost_graph, fm_sampling_explorer.fbFM.path, min_cost=barlims[0], max_cost=barlims[1])
    tempframe.append(ax1.plot(fm_sampling_explorer.X[:,0], fm_sampling_explorer.X[:,1], 'rx', mew=1)[0])
    tempframe.append(ax1.plot(fm_bestX[0], fm_bestX[1], 'wo', mfc='none', ms=12, mew=2, mec='w')[0])
    fig1.savefig(VID_DIR+'new_estimate_withpath.png', bbox_inches='tight', transparent=True)


plt.show()