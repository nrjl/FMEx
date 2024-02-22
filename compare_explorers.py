import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import time
import math
import bfm_explorer
from fm_tools import fast_marcher, fm_graphtools, fm_plottools
NUM_SAMPLES = 20
            
gridsize = [100, 101]
random.seed(1)

def explore_cost_function(a, b):
    cost = 3 
    cost += 10*math.exp(-math.sqrt((a-40)**2 + (b-40)**2)/16)
    cost += 7*math.exp(-math.sqrt((a-10)**2 + (b-90)**2)/12)
    cost += 4*math.exp(-math.sqrt((a-80)**2 + (b-60)**2)/32)
    cost += 7*math.exp(-math.sqrt((a+20)**2 + (b-50)**2)/32)
    cost += 7*math.exp(-math.sqrt((a-120)**2 + (b-50)**2)/32)
    cost += 12*math.exp(-math.sqrt((a-80)**2 + (b-20)**2)/8)
    cost += 5*math.exp(-math.sqrt((a-60)**2 + (b-80)**2)/10)
    cost += 3*math.exp(-math.sqrt((a-90)**2 + (b-90)**2)/20)
    return cost
    
def sample_cost_fun(cf, x):
    y = cf(x[0], x[1]) + random.normalvariate(0, 0.25)
    return y

def calc_true_path_cost(cost_fun, path):
    true_cost = 0
    for i in range(len(path)):
        true_cost += cost_fun(path[i][0], path[i][1])
    return true_cost

def calc_est_path_cost(gp_model, mean_val, path):
    true_cost,var_cost = gp_model.predict(np.asarray(path))
    true_cost += mean_val
    return true_cost.sum(), var_cost.sum()
    
def check_row(mat, row):
    # Check if a row is equal to an existing row in a matrix
    for ii in range(mat.shape[0]):
        this_row = True
        for jj in range(mat.shape[1]):
            this_row *= (mat[ii,jj] == row[jj])
        if this_row:
            return True
    return False

def plot_updates(ax, models):        
    axax = [ax[0][1], ax[1][0], ax[1][1]]
    graph_frame = []
    for i in range(3):
        tempframe, barlims = fm_plottools.draw_grid(axax[i], models[i].GP_cost_graph, models[i].fbFM.path, 19)
        tempframe.append(axax[i].plot(models[i].X[:,0], models[i].X[:,1], 'rx')[0])
        graph_frame.extend(tempframe)
    return graph_frame

true_g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], explore_cost_function)
explorer_cost = bfm_explorer.mat_cost_function(true_g, explore_cost_function)
true_g.cost_fun = explorer_cost.calc_cost

true_g.obstacles = [] #fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 10, 25)
start_node = (3,5)
end_node = (97, 97)

print('Creating true field fast marcher... ', end=' ')
# Create search for true field
tFM = fast_marcher.FullBiFastMarcher(true_g)
tFM.set_start(start_node)
tFM.set_goal(end_node)
tFM.search()
tFM.pull_path()
best_path_cost = calc_true_path_cost(explore_cost_function, tFM.path)
print('done. Best path cost = {0}'.format(best_path_cost))

fig1, ax1 = plt.subplots(2, 2, sharex=True, sharey=True)
ax1[0][0].set_title("True cost field")
ax1[0][1].set_title("Estimated cost - random sampling")
ax1[1][0].set_title("Estimated cost - max variance sampling")
ax1[1][1].set_title("Estimated cost - FM sampling")
fm_plottools.draw_grid(ax1[0][0], true_g, tFM.path, 19)
#fm_plottools.draw_fbfmcost(ax1[0][0], true_g, tFM.path_cost, tFM.path, 1000, 1400)

print('Creating fast marching explorers... ', end=' ')
# Create initial GP
mean_value = 3;

X = np.array([[3, 5]])
Xshape = X.shape
Y = np.zeros((Xshape[0], 1))
for ii in range(Xshape[0]):
    Y[ii] = explore_cost_function(X[ii,0], X[ii,1]) + random.normalvariate(0, 0.5)

# Create BiFM explorer objects for each sampling strategy
random_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
maxvar_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
fm_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
print('done.')

## UPDATES!
#cost_update = square_cost_modifier(g, 60, 80, 10, 30, -3)
test_gridx = list(range(2, true_g.width, 12)); lx = len(test_gridx)
test_gridy = list(range(2, true_g.height, 12)); ly = len(test_gridy)
delta_costs = [-1, 1]; ld = len(delta_costs)

NUM_TESTS = lx*ly*ld

search_time = np.zeros(NUM_SAMPLES, float)

true_path_cost = np.zeros((3, NUM_SAMPLES), float)
est_path_cost = np.zeros((3, NUM_SAMPLES), float)
est_path_var = np.zeros((3, NUM_SAMPLES), float)

video_frames = []

for ii in range(NUM_SAMPLES):
    
    t0 = time.time()   
    
    # Random sampler:
    tx = random.choice(test_gridx)
    ty = random.choice(test_gridy)
    while check_row(random_sampling_explorer.X, [tx,ty]):
        tx = random.choice(test_gridx)
        ty = random.choice(test_gridy)
    random_sampling_explorer.add_observation([tx,ty], sample_cost_fun(explore_cost_function, [tx,ty]))
    true_path_cost[0,ii] = calc_true_path_cost(explore_cost_function, random_sampling_explorer.fbFM.path)
    est_path_cost[0,ii],est_path_var[0,ii] = calc_est_path_cost(random_sampling_explorer.GP_model, mean_value, random_sampling_explorer.fbFM.path)
    
    # Max var and fm samplers
    max_var = 0
    fm_best_cost = -1
    
    for tx in test_gridx:
        for ty in test_gridy:
            # Max var
            if not check_row(maxvar_sampling_explorer.X, [tx,ty]):
                temp_var = maxvar_sampling_explorer.varYfull[ty*gridsize[0]+tx]
                if temp_var > max_var:
                    max_var = temp_var
                    maxvar_bestX = [tx,ty]
           
            if not check_row(fm_sampling_explorer.X, [tx,ty]):
                current_value = 0
                for td in delta_costs:
                    stdY = math.sqrt(fm_sampling_explorer.varYfull[ty*gridsize[0]+tx])
                    cost_update =fm_graphtools.polynomial_cost_modifier(fm_sampling_explorer.GP_cost_graph, tx, ty, 15, td*stdY)
                    current_value += fm_sampling_explorer.cost_update(cost_update)
                if fm_best_cost == -1 or (current_value < fm_best_cost):
                    fm_best_cost = current_value
                    fm_bestX = [tx,ty]
                    
    maxvar_sampling_explorer.add_observation(maxvar_bestX, sample_cost_fun(explore_cost_function, maxvar_bestX))
    true_path_cost[1,ii] = calc_true_path_cost(explore_cost_function, maxvar_sampling_explorer.fbFM.path)
    est_path_cost[1,ii],est_path_var[1,ii] = calc_est_path_cost(maxvar_sampling_explorer.GP_model, mean_value, maxvar_sampling_explorer.fbFM.path)
    
    fm_sampling_explorer.add_observation(fm_bestX, sample_cost_fun(explore_cost_function, fm_bestX))
    true_path_cost[2,ii] = calc_true_path_cost(explore_cost_function, fm_sampling_explorer.fbFM.path)
    est_path_cost[2,ii],est_path_var[2,ii] = calc_est_path_cost(fm_sampling_explorer.GP_model, mean_value, fm_sampling_explorer.fbFM.path)
    
    video_frames.append(plot_updates(ax1, [random_sampling_explorer, maxvar_sampling_explorer, fm_sampling_explorer]))            
    print("Iteration {0} path costs - Random: {1}, MaxVar: {2}, FM: {3}".format(ii, true_path_cost[0,ii], true_path_cost[1,ii], true_path_cost[2,ii]))
    search_time[ii] = time.time()-t0
    
ani1 = animation.ArtistAnimation(fig1, video_frames, interval=500, repeat_delay=0)

fig_costs, ax_costs = plt.subplots()
ax_costs.plot([1, NUM_SAMPLES], [1, 1], 'k--', label='Best path cost')
ax_costs.plot(list(range(1, NUM_SAMPLES+1)), [x/best_path_cost for x in true_path_cost[0,:]], 'b-', label='Random sampling')
ax_costs.plot(list(range(1, NUM_SAMPLES+1)), [x/best_path_cost for x in true_path_cost[1,:]], 'r-', label='Max variance sampling')
ax_costs.plot(list(range(1, NUM_SAMPLES+1)), [x/best_path_cost for x in true_path_cost[2,:]], 'g-', label='FMCost sampling')
ax_costs.legend(loc=0)
ax_costs.set_xlabel('Samples')
ax_costs.set_ylabel('Path cost (normalized against best path cost)')



plt.show()