import numpy as np
import random
import time
import math
import bfm_explorer
import fast_marcher
import fm_graphtools
import fm_plottools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import statrun_plots

plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

SAMPLING_VIDEOS = False  # Turn on video frames for each sample
OBSTACLES_ON = True      # Obstacles on
TIMED_PLOTS = False     # Output (some) frames as pdf images

NUM_STATRUNS = 60       # Number of statistical runs (random maps) 
NUM_SAMPLES = 50         # Number of sample points in each run
            
gridsize = [100, 100]    # Grid size
SEED_NUM = 9
random.seed(SEED_NUM)           # Random seed

mean_value = 5.0           # Mean value of the field for GP
num_blobs = 30             # Number of blobs in each field
peak_range = [-3.0,8.0]    # Low and high peak values for map blobs
spread_range = [5,12]      # Low and high spread distances for map blobs
poly_update_size = 15

GP_l=10.0; GP_sv=15.0; GP_sn=0.1
OBS_STD = 0.05

delta_costs = [-1.0, 0.5]   # How many stds above and below for FM sampling
delta_costs2 = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
n_random_samples = 6

nmethods = 6
plot_timer = 10
ex_plot_index = [0,1,3]
labels = ['Random', 'Max Variance', 'LCB', 'FMEx $1\sigma$', 'FMEx $3\sigma$', 'FMEx MC{0:d}'.format(n_random_samples)]

num_obstacles = 40       # Total number of obstacles
obstacle_size = 10       # Obstacle size

DATA_DIR = '/home/nick/Dropbox/work/FastMarching/FMEx/data/'
VID_DIR = '/home/nick/Dropbox/work/FastMarching/FMEx/vid/'
FIG_DIR = '/home/nick/Dropbox/work/FastMarching/FMEx/fig/'

my_blobs = [[30, 20, 16, 10], [60, 40,  6, 15], [10, 40, 32, 12], [60, 5, 15, 9],
            [25, 35,  8, 15], [70, 30, 12, 12], [80, 40, 16, 15], [5, 60, 30, 7],
            [30, 70, 32, 18], [90, 10, 20, 15]]

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2)
    return y
    
def explore_cost_function(a, b, blobs):
    # blobs are defined by [xloc, yloc, spread, peakval]
    # blobs = my_blobs
    cost = mean_value
    for i in range(np.shape(blobs)[0]):
        cost += blobs[i][3]*math.exp(-math.sqrt((a-blobs[i][0])**2 + (b-blobs[i][1])**2)/blobs[i][2])
    return cost
    
def sample_cost_fun(cf, x, blobs):
    y = cf(x[0], x[1], blobs) + random.normalvariate(0, OBS_STD)
    return max(0, y)

def sample_value(fm, pc, x, y, delta_costs):
    std = math.sqrt(fm.GP_cost_graph.var_fun(x,y))
    cc = 0
    for dc in delta_costs:
        cc += fm.cost_update(pc.set_update(x, y, dc*std))
    return cc
    
def sample_value_new(fm, pc, x, y, delta_costs):
    std = math.sqrt(fm.GP_cost_graph.var_fun(x,y))
    path_cost = 0
    for dc in delta_costs:
        weight = normpdf(dc, 0.0, 1.0)
        path_cost += weight*fm.cost_update_new(pc.set_update(x, y, dc*std))
    return path_cost
    
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
    
def check_row(mat, row):
    return np.equal(mat,row).all(1).any()

def create_cost_plot(graph=None):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    if graph != None:
        for axn in ax.flat:
            axn.set_aspect('equal')
            axn.tick_params(labelbottom='on',labeltop='off')
            axn.set_xlim(-0.5, graph.width-.5)
            axn.set_ylim(-0.5, graph.height-0.5)            
    else:
        for axn in ax.flat:
            axn.autoscale(tight=True)
 
    ax[0][0].set_title("True cost field")
    ax[0][1].set_title("Random sampling")
    ax[1][0].set_title("Max Variance sampling")
    ax[1][1].set_title("FMEx")
    #ax[1][2].set_title("Estimated cost - FM sampling2")
    return fig,ax

def plot_final_paths(ax, true_g, true_path, models):        
    axax = [ax[0][1], ax[1][0], ax[1][1]]#, ax[1][2]]
    graph_frame = []
    #if not ax[0][0].lines:
    tempframe, barlims = fm_plottools.draw_grid(ax[0][0], true_g, true_path)
    graph_frame.extend(tempframe)
    for i in range(len(models)):
        tempframe, barlims = fm_plottools.draw_grid(axax[i], models[i].GP_cost_graph, path=models[i].fbFM.path, max_cost=barlims[1],min_cost=barlims[0] )
        tempframe.append(axax[i].plot(models[i].X[:,0], models[i].X[:,1], 'rx')[0])
        graph_frame.extend(tempframe)
    return graph_frame

def fmex_constructor(gridsize, start_node, end_node, X, Y, mean_value, obstacles):
    fmex = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value=mean_value, obs=obstacles,
        GP_l=GP_l, GP_sv=GP_sv, GP_sn=GP_sn)
    fmex.search()
    return fmex

best_path_cost = np.zeros((1, NUM_STATRUNS), float)
true_path_cost = np.zeros((NUM_STATRUNS, nmethods, NUM_SAMPLES), float)
est_path_cost = np.zeros((NUM_STATRUNS, nmethods, NUM_SAMPLES), float)
est_path_var = np.zeros((NUM_STATRUNS, nmethods, NUM_SAMPLES), float)
sample_times = np.zeros((NUM_STATRUNS, NUM_SAMPLES), float)

temp_g = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1])

if SAMPLING_VIDEOS:
    fig_s,ax_s = create_cost_plot(temp_g)
if TIMED_PLOTS:
    fig_t,ax_t = create_cost_plot(temp_g)
      
fig_v,ax_v = create_cost_plot(temp_g)

video_frames=[]
nowstr = time.strftime("%Y_%m_%d-%H_%M")

fh = open(DATA_DIR+nowstr+"_summary.txt", "w" )
fh.write("Obstacles: {0}\n".format(OBSTACLES_ON))
if OBSTACLES_ON:
    fh.write("\tNumber: {0}, Size: {1}\n".format(num_obstacles,obstacle_size ))
fh.write("Number of statruns: {0}\n".format(NUM_STATRUNS))
fh.write("Number of samples: {0}\n".format(NUM_SAMPLES))
fh.write("Gridsize: {0}\n".format(gridsize))
fh.write("Random seed: {0}\n".format(SEED_NUM))
fh.write("Mean value: {0}\n".format(mean_value))
fh.write("Map blobs: {0}, Peak range: {1}, Spread range: {2}\n".format(num_blobs, peak_range, spread_range))
fh.write("Delta costs: {0}, {1}\n".format(delta_costs, delta_costs2))
fh.write("GP: l={0}, s_v={1}, s_n={2}\n".format(GP_l, GP_sv, GP_sn))
fh.close()
print 'NOWSTR: {0}'.format(nowstr)

jj= 0

def random_sampler(explorer, rand_samples, *arg):
    return rand_samples[0]
    
def maxvar_sampler(explorer, rand_samples, *arg):
    tvar = [explorer.GP_cost_graph.var_fun(x[0],x[1]) for x in rand_samples]
    return rand_samples[np.argmax(tvar)] 

def lcb_sampler(explorer, rand_samples,*arg):
    tvar = [explorer.GP_cost_graph.cost_fun(x[0],x[1])-math.sqrt(explorer.GP_cost_graph.var_fun(x[0],x[1])) for x in rand_samples]
    return rand_samples[np.argmin(tvar)]      
    
def fmex_sampler(explorer, rand_samples, cost_obj, delta_costs):
    E_path_cost = [sample_value_new(explorer, cost_obj, x[0],x[1], delta_costs) for x in rand_samples]
    return rand_samples[np.argmin(E_path_cost)]       


#for jj in range(NUM_STATRUNS):
while jj < NUM_STATRUNS:
    plt_tmr = plot_timer-1
    t0 = time.time()  
    sampling_frames=[]
    cblobs = []
    for ii in range(num_blobs):
        cblobs.append([random.uniform(-10,gridsize[0]+10), random.uniform(-10,gridsize[1]+10), 
            random.uniform(spread_range[0], spread_range[1]), random.uniform(peak_range[0], peak_range[1])])
    
    true_g = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], explore_cost_function, [])
    
    if OBSTACLES_ON:
        true_g.update_obstacles(fm_plottools.generate_obstacles(true_g, num_obstacles, obstacle_size))
    explorer_cost = bfm_explorer.mat_cost_function(true_g, explore_cost_function, cblobs)
    true_g.cost_fun = explorer_cost.calc_cost
        
    poly_cost_obj = fm_graphtools.polynomial_precompute_cost_modifier(true_g, poly_update_size, min_val=0.001)
        
    start_node = (3,3)
    while start_node in true_g.obstacles:
        start_node = (start_node[0]+1, start_node[1])
    end_node = (97, 97)
    while end_node in true_g.obstacles:
        end_node = (end_node[0]-1, end_node[1])
    
    # Create search for true field
    tFM = fast_marcher.FullBiFastMarcher(true_g)
    tFM.set_start(start_node)
    tFM.set_goal(end_node)
    try:
        tFM.search()
    except KeyError:
        print "No map solution, moving to next map".format(jj)
        continue
    tFM.pull_path()
    best_path_cost[0][jj] = calc_true_path_cost(explore_cost_function, tFM.path, cblobs)
        
    # Create initial GP    
    X = np.array([[3, 5]])
    Xshape = X.shape
    Y = np.zeros((Xshape[0], 1))
    for ii in range(Xshape[0]):
        Y[ii] = sample_cost_fun(explore_cost_function, X[ii,:], cblobs)
    
    # Create BiFM explorer objects for each sampling strategy
    t0 = time.time()
    explorers = [fmex_constructor(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles) for nn in range(nmethods)]
    #print "Construction took {0}s".format(time.time()-t0)
    
    ## UPDATES!
    
    for ii in range(NUM_SAMPLES):        
       
        rand_samples = []
        while len(rand_samples) < 50:
            tx = random.choice(range(gridsize[0]))
            ty = random.choice(range(gridsize[1]))
            if ((tx,ty) not in true_g.obstacles):
                rand_samples.append((tx,ty))
        
        bestX = np.zeros((nmethods,2))

        # Run samplers:
        bestX[0,:] = random_sampler(explorers[0], rand_samples)
        bestX[1,:] = maxvar_sampler(explorers[1], rand_samples)
        bestX[2,:] = lcb_sampler(explorers[2], rand_samples)
        bestX[3,:] = fmex_sampler(explorers[3], rand_samples, poly_cost_obj, delta_costs)
        bestX[4,:] = fmex_sampler(explorers[4], rand_samples, poly_cost_obj, delta_costs2)
        bestX[5,:] = fmex_sampler(explorers[5], rand_samples, poly_cost_obj, np.random.normal(loc=0, scale=1.0, size=n_random_samples))
           
        for dex, explorer in enumerate(explorers):
            explorer.add_observation([bestX[dex,:]], [[sample_cost_fun(explore_cost_function, bestX[dex,:], cblobs)]])
            true_path_cost[jj,dex,ii] = calc_true_path_cost(explore_cost_function, explorer.fbFM.path, cblobs)
            est_path_cost[jj,dex,ii],est_path_var[jj,dex,ii] = calc_est_path_cost(explorer.GP_model, mean_value, explorer.fbFM.path)

        #print('Update map time: {0}s'.format(time.time()-ts))
        ts = time.time()
        
        
        if SAMPLING_VIDEOS:
            sampling_frames.append(plot_final_paths(ax_s, true_g, tFM.path, [explorers[nn] for nn in ex_plot_index]))

        if TIMED_PLOTS and ii == plt_tmr:
            tf = plot_final_paths(ax_t, true_g, tFM.path, [explorers[nn] for nn in ex_plot_index])
            fig_t.savefig(FIG_DIR+nowstr+'T{0}S{1}.pdf'.format(jj,ii+1), bbox_inches='tight')
            plt_tmr += plot_timer
            for item in tf:
                item.remove()
        #print('Plotting time: {0}s'.format(time.time()-ts))
    
    if SAMPLING_VIDEOS:
        for exframe in range(5):
            sampling_frames.append(sampling_frames[-1])
        ani_sampling = animation.ArtistAnimation(fig_s, sampling_frames, interval=100, repeat_delay=0)
        ani_sampling.save('{0}{1}S{2:02d}.mp4'.format(VID_DIR, nowstr, jj), writer = 'avconv', fps=2, bitrate=1500)        
    best_path_cost[0][jj] = min([best_path_cost[0][jj], true_path_cost[jj,0,-1], true_path_cost[jj,1,-1], true_path_cost[jj,3,-1]])
    video_frames.append(plot_final_paths(ax_v, true_g, tFM.path, [explorers[nn] for nn in ex_plot_index]))
    if sampling_frames: 
        for item in sampling_frames[-1]:
            item.remove()
    print "STAT RUN {k}: Best: {cB:.2f},".format(k=jj, cB=best_path_cost[0][jj]),
    for nn in range(nmethods):
        print "{n}:{cB:.2f},".format(n=labels[nn], cB=np.mean(true_path_cost[jj,nn,:])),
    print "Total  {t:.2f}s".format(t=time.time()-t0)
    jj += 1

fh = open(DATA_DIR+nowstr+".p", "wb" )
pickle.dump(best_path_cost, fh)
pickle.dump(true_path_cost, fh)
pickle.dump(est_path_cost, fh)
pickle.dump(est_path_var, fh)
fh.close()


ani1 = animation.ArtistAnimation(fig_v, video_frames, interval=1000, repeat_delay=0)
ani1.save(VID_DIR+nowstr+'.mp4', writer = 'avconv', fps=1, bitrate=1500)


fig1, fig2, fig3, fig4 = statrun_plots.make_plots(best_path_cost, true_path_cost, est_path_cost, labels, comparison=3, cols=['cornflowerblue', 'green', 'firebrick', 'orange', 'purple', 'darkslateblue'])
fig1.savefig(FIG_DIR+nowstr+'C.pdf', bbox_inches='tight')
fig2.savefig(FIG_DIR+nowstr+'.pdf', bbox_inches='tight')
fig3.savefig(FIG_DIR+nowstr+'L.pdf', bbox_inches='tight')
fig4.savefig(FIG_DIR+nowstr+'E.pdf', bbox_inches='tight')
plt.show()


#ax2 = plt.subplot()
#cols = ['b', 'g', 'k']
#labels = ['Random', 'Max Variance', 'Fast March']
#for i in range(3):
#    tempdata = [np.divide(true_path_cost[:,i,j], best_path_cost)-1 for j in np.append(np.arange(0, NUM_SAMPLES, 3), NUM_SAMPLES-1)]
#    pos = np.append(np.arange(0,NUM_SAMPLES*4,12)+i+1, NUM_SAMPLES*4-2)
#    ax2.boxplot(tempdata, positions=pos, medianprops={'color':'r'}, 
#        showbox=True, boxprops={'color':cols[i]}, notch=True, showcaps=False, showfliers=False, 
#        whis=0, whiskerprops={'color':cols[i]}, flierprops={'color':cols[i]}, bootstrap=5000)  #, 
#    tempdata = [np.divide(true_path_cost[:,i,j], best_path_cost)-1 for j in range(0, NUM_SAMPLES)]    
#    ax2.plot(np.arange(0,NUM_SAMPLES*4,4)+i+1, np.median(tempdata, axis=2), cols[i], label=labels[i])
#ax2.set_xlim(0, 4*NUM_SAMPLES)
#ax2.set_xticks(np.arange(0,NUM_SAMPLES*4,4)+2)
#ax2.set_xticklabels(np.arange(NUM_SAMPLES)+1)
#ax2.set_yscale('log')
#ax2.legend()
#
#plt.show()