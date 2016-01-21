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


#from explorer_statrun import explore_cost_function, sample_cost_fun, calc_true_path_cost, calc_est_path_cost
def explore_cost_function(a, b, blobs):
    # blobs are defined by [xloc, yloc, spread, peakval]
    # blobs = my_blobs
    cost = mean_value
    for i in range(np.shape(blobs)[0]):
        cost += blobs[i][3]*math.exp(-math.sqrt((a-blobs[i][0])**2 + (b-blobs[i][1])**2)/blobs[i][2])
    return cost
    
def sample_cost_fun(cf, x, blobs):
    y = cf(x[0], x[1], blobs) + random.normalvariate(0, 0.8)
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

MAKE_VIDEO = True
MAKE_PICTURES=False
gridsize = [100, 80]
mean_value = 3
NUM_SAMPLES = 1
OBSTACLES_ON = True
num_obstacles = 50
obstacle_size = 10
delta_costs = [-1, 1]
random.seed(2)

num_blobs = 35           # Number of blobs in each field
peak_range = [5,15]      # Low and high peak values for map blobs
spread_range = [5,12]    # Low and high spread distances for map blobs

VID_DIR = '/home/nick/Dropbox/work/FastMarching/paper/icra/video/'

cblobs = []
for ii in range(num_blobs):
    cblobs.append([random.uniform(-10,gridsize[0]+10), random.uniform(-10,gridsize[1]+10), 
        random.uniform(spread_range[0], spread_range[1]), random.uniform(peak_range[0], peak_range[1])])
            
true_g = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], explore_cost_function, [])
explorer_cost = bfm_explorer.mat_cost_function(true_g, explore_cost_function, cblobs)
true_g.cost_fun = explorer_cost.calc_cost
if OBSTACLES_ON:
    true_g.update_obstacles(fm_plottools.generate_obstacles(gridsize[0], gridsize[1], num_obstacles, obstacle_size))

poly_cost_obj = fm_graphtools.poly_cost(true_g, 13)
        
start_node = (3,5)
end_node = (97, 77)
while start_node in true_g.obstacles:
    start_node = (start_node[0]+1, start_node[1])
while end_node in true_g.obstacles:
    end_node = (end_node[0]-1, end_node[1])

X = np.array([[3, 5], [30,70], [90,60], [44,50], [1,34], [72,8]])
Xshape = X.shape
Y = np.zeros((Xshape[0], 1))
for ii in range(Xshape[0]):
    Y[ii] = sample_cost_fun(explore_cost_function, X[ii,:], cblobs)


fm_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
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
    tempframe.append(ax1.plot(53, 30, 'wo', mfc='none', ms=12, mew=2, mec='w')[0])
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
    for whitmans_sampler in range(100):
        tx = random.choice(range(gridsize[0]))
        ty = random.choice(range(gridsize[1]))
        while (tx,ty) in true_g.obstacles:
            tx = random.choice(range(gridsize[0]))
            ty = random.choice(range(gridsize[1]))            

        current_value = 0
        for td in delta_costs:
            stdY = math.sqrt(fm_sampling_explorer.varYfull[ty*gridsize[0]+tx])
            cost_update = poly_cost_obj.set_update(tx, ty, td*stdY)
            current_value += fm_sampling_explorer.cost_update_new(cost_update, [tx,ty])
        if fm_best_cost == -1 or (current_value < fm_best_cost):
            fm_best_cost = current_value
            fm_bestX = [tx,ty]
    print "Sample selection time: {0:.4f}".format(time.time()-ts)                                        
        
    fm_sampling_explorer.add_observation(fm_bestX, sample_cost_fun(explore_cost_function, fm_bestX, cblobs))
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