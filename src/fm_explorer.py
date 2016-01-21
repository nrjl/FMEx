import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import time
import math
import copy
import GPy
# from dijkstra_search import dijkstra_search, pull_path
import fast_marcher
import fm_graphtools
import fm_plottools

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

VID_DIR = '/home/nick/Dropbox/work/FastMarching/explore_vid/'
PLOT_UPDATES = False  # This will overwrite to one sample for plotting purposes
NUM_SAMPLES = 30
RANDOM_SAMPLER = True # Add model that samples randomly
MAX_VARIANCE_SAMPLER = True # Add model that samples max variance

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

def calc_true_path_cost(cost_fun, path):
    true_cost = 0
    for i in range(len(path)):
        true_cost += cost_fun(path[i][0], path[i][1])
    return true_cost

def calc_est_path_cost(gp_model, mean_val, path):
    true_cost,var_cost = gp_model.predict(np.asarray(path))
    true_cost += mean_val
    return true_cost.sum(), var_cost.sum()
    
def zero_fun(a,b):
    return 0
    
def check_row(mat, row):
    # Check if a row is equal to an existing row in a matrix
    for ii in range(mat.shape[0]):
        this_row = True
        for jj in range(mat.shape[1]):
            this_row *= (mat[ii,jj] == row[jj])
        if this_row:
            return True
    return False

class mat_cost_function:
    def __init__(self, graph, cost_fun=zero_fun):
        self.mat = np.zeros((graph.width, graph.height))
        for x in range(graph.width):
            for y in range(graph.height):
                self.mat[x,y] = cost_fun(x,y)
    
    def calc_cost(self,a,b):
        return self.mat[a,b]
            
gridsize = [100, 100]
random.seed(1)

true_g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], explore_cost_function)
explorer_cost = mat_cost_function(true_g, explore_cost_function)
true_g.cost_fun = explorer_cost.calc_cost

true_g.obstacles = [] #fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 10, 25)
start_node = (3,3)
end_node = (97, 97)


# Create search for true field
tFM = fast_marcher.FullBiFastMarcher(true_g)
tFM.set_start(start_node)
tFM.set_goal(end_node)
t0 = time.time()
tFM.search()
tFM.pull_path()

fig1, ax1 = plt.subplots(2, 2, sharex=True, sharey=True)
ax1[0][0].set_title("True cost field")
ax1[0][1].set_title("Estimated cost field")
ax1[1][0].set_title("True path cost")
ax1[1][1].set_title("Estimated path cost")

fm_plottools.draw_fbfmcost(ax1[1][0], true_g, tFM.path_cost, tFM.path, 1000, 1400)

# Create initial GP
mean_value = 3;
GPg = copy.copy(true_g)
#X = 100.0*np.random.rand(100,2)
if PLOT_UPDATES:
    X = np.array([[3, 3], [14, 14], [14, 38], [38, 14], [38, 38], [62, 14], [2, 62], [86, 14], [26, 74], [74, 38]])
else:
    X = np.array([[3, 3]])
Xshape = X.shape
Y = np.zeros((Xshape[0], 1))
for ii in range(Xshape[0]):
    Y[ii] = explore_cost_function(X[ii,0], X[ii,1]) + random.normalvariate(0, 0.5)
# define kernel
ker = GPy.kern.RBF(2)
#ker = GPy.kern.Poly(2)
# create simple GP model
m = GPy.models.GPRegression(X,Y-mean_value,ker)
m.kern.lengthscale = 15
m.kern.variance = 20
m.Gaussian_noise.variance = 1

Xtemp, Ytemp = np.mgrid[0:100, 0:100]
Xfull = np.vstack([Xtemp.ravel(), Ytemp.ravel()]).transpose()
cmodel = mat_cost_function(GPg)
Yfull, varYfull = m.predict(Xfull)
Yfull += mean_value
cmodel.mat = np.reshape(Yfull, (100, 100))
GPg.cost_fun = cmodel.calc_cost

estimate_frames=[]

fm_plottools.draw_grid(ax1[0][0], true_g, [], 19)
graph_frame, barlims = fm_plottools.draw_grid(ax1[0][1], GPg, [], 19)
graph_frame.append(ax1[0][1].plot(X[:,0], X[:,1], 'rx')[0])
#cbar = fig1.colorbar(graph_frame[0])

fbFM = fast_marcher.FullBiFastMarcher(GPg)
fbFM.set_start(start_node)
fbFM.set_goal(end_node)

# Initial search
fbFM.search()
fbFM.pull_path()
#figfbFM, axfbFM = fm_plottools.init_fig()
fbFM_frames = []
tempframe, tempbar = fm_plottools.draw_fbfmcost(ax1[1][1], GPg, fbFM.path_cost, fbFM.path, 1000, 1400)
tempframe.extend(graph_frame)
fbFM_frames.append(tempframe)

if PLOT_UPDATES:
    sg_frames = []
    fig_sg, ax_sg = fm_plottools.init_fig()
    fbFM.FastMarcherSG.set_plots(sg_frames, ax_sg)
    fbFM.FastMarcherSG.set_plot_costs(0, 20)
    gs_frames = []
    fig_gs, ax_gs = fm_plottools.init_fig()
    fbFM.FastMarcherGS.set_plots(gs_frames, ax_gs)
    fbFM.FastMarcherGS.set_plot_costs(0, 20)
    up_frames = []
    fig_up, ax_up = fm_plottools.init_fig()
    fbFM.set_plots(up_frames, ax_up)
    fbFM.set_plot_costs(0, 20)

## UPDATES!
#cost_update = square_cost_modifier(g, 60, 80, 10, 30, -3)
test_gridx = range(2, 100, 12); lx = len(test_gridx)
test_gridy = range(2, 100, 12); ly = len(test_gridy)
delta_costs = [-1, 1]; ld = len(delta_costs)

NUM_TESTS = lx*ly*ld

if PLOT_UPDATES:
    NUM_SAMPLES = 1

search_time = np.zeros(NUM_SAMPLES, float)

true_path_cost = []
est_path_cost = []
est_path_var = []

for ii in range(NUM_SAMPLES):
    
    t0 = time.time()   
    best_cost = -1
    
    for tx in test_gridx:
        for ty in test_gridy:
            if not check_row(X, [tx,ty]):
                current_value = 0
                for td in delta_costs:
                    stdY = math.sqrt(varYfull[tx*GPg.width+ty])
                    cost_update =fm_graphtools.polynomial_cost_modifier(GPg, tx, ty, 15, td*stdY)
                    fbFM.update(cost_update)
                    current_value += fbFM.updated_min_path_cost
                if best_cost == -1 or (current_value < best_cost):
                    best_cost = current_value
                    bestX = [tx,ty]
    
    print "Point selected: ({0}, {1}). Estimation time = {2} sec.".format(bestX[0], bestX[1], time.time()-t0)
    search_time[ii] = time.time()-t0
    
    # update GP with best observation
    X = np.append(X, [bestX], axis=0)
    Y = np.append(Y, [[explore_cost_function(bestX[0], bestX[1]) + random.normalvariate(0, 0.5)]], axis=0)
    m.set_XY(X, Y-mean_value)

    Yfull, varYfull = m.predict(Xfull)
    Yfull += mean_value
    cmodel.mat = np.reshape(Yfull, (100, 100))
    GPg.cost_fun = cmodel.calc_cost
    GPg.clear_delta_costs()
    
    fbFM.set_graph(GPg)
    fbFM.set_start(start_node)
    fbFM.set_goal(end_node)
    fbFM.search()
    fbFM.pull_path()
    tempframe, barlims = fm_plottools.draw_fbfmcost(ax1[1][1], GPg, fbFM.path_cost, fbFM.path, 1000, 1400)
    fbFM_frames.append(tempframe)  
    
    graph_frame, barlims = fm_plottools.draw_grid(ax1[0][1], GPg, fbFM.path, 19)
    graph_frame.append(ax1[0][1].plot(X[:,0], X[:,1], 'rx')[0])
    tempframe.extend(graph_frame)
    fbFM_frames.append(tempframe)
    
    # Calculate estimated and true path costs
    true_path_cost.append(calc_true_path_cost(explore_cost_function, fbFM.path))
    t_c, t_var = calc_est_path_cost(m, mean_value, fbFM.path)
    est_path_cost.append(t_c)
    est_path_var.append(t_var)
      
#if fbFM.image_frames != 0:
#    ani3 = animation.ArtistAnimation(fig3, fbFM.image_frames, interval=10, repeat_delay=2000)
for ii in range(5):
    fbFM_frames.append(tempframe)
        
ani1 = animation.ArtistAnimation(fig1, fbFM_frames, interval=100, repeat_delay=0)

if PLOT_UPDATES:
    for ii in range(10): sg_frames.append(sg_frames[-1])
    for ii in range(10): gs_frames.append(gs_frames[-1])
    aniSG = animation.ArtistAnimation(fig_sg, sg_frames, interval=10, repeat_delay = 1000)
    aniGS = animation.ArtistAnimation(fig_gs, gs_frames, interval=10, repeat_delay = 1000)
    aniUP = animation.ArtistAnimation(fig_up, up_frames, interval=10, repeat_delay = 1000)
    #aniSG.save(VID_DIR+'forward_search.mp4', writer = 'avconv', fps=25, bitrate=1500)
    #aniGS.save(VID_DIR+'backward_search.mp4', writer = 'avconv', fps=25, bitrate=1500)
    #aniUP.save(VID_DIR+'updates.mp4', writer = 'avconv', fps=50, bitrate=1500)
            
#anifbFM = animation.ArtistAnimation(figfbFM, fbFM_frames, interval=100, repeat_delay=2000)

# fig1.set_size_inches(9.5,8.5)
# ani1.save(VID_DIR+'field_estimate.mp4', writer = 'avconv', fps=2, bitrate=1500)
# anifbFM.save(VID_DIR+'path_cost.mp4', writer = 'avconv', fps=2, bitrate=1500)

best_cost = calc_true_path_cost(explore_cost_function, tFM.path)
fig_costs, ax_costs = fm_plottools.init_fig()
l0,=ax_costs.plot([1, NUM_SAMPLES], [1, 1], 'k--', label='Best path cost')
l1,=ax_costs.plot(range(1, NUM_SAMPLES+1), [x/best_cost for x in true_path_cost], 'b-', label='True path cost, biFM')
l2,=ax_costs.plot(range(1, NUM_SAMPLES+1), [x/best_cost for x in est_path_cost], 'r-', label='Estimated path cost, biFM')
#l3,=ax_costs.plot(range(1, NUM_SAMPLES+1), est_path_cost+np.sqrt(est_path_var), 'r--', label='Estimated path uncertainty ($1-\sigma$)')
#ax_costs.plot(range(1, NUM_SAMPLES+1), est_path_cost-np.sqrt(est_path_var), 'r--')
ax_costs.set_aspect('auto', 'datalim')
ax_costs.autoscale(enable=True, axis='both',tight='False')
ax_costs.legend(loc=4)
ax_costs.set_xlabel('Samples')
ax_costs.set_ylabel('Path cost (normalized)')

plt.show()