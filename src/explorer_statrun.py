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

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

SAMPLING_VIDEOS = True  # Turn on video frames for each sample
OBSTACLES_ON = True      # Obstacles on
TIMED_PLOTS = False     # Output (some) frames as pdf images

NUM_STATRUNS = 10       # Number of statistical runs (random maps) 
NUM_SAMPLES = 30         # Number of sample points in each run
            
gridsize = [100, 100]    # Grid size
SEED_NUM = 9
random.seed(SEED_NUM)           # Random seed

mean_value = 3           # Mean value of the field for GP
num_blobs = 25           # Number of blobs in each field
peak_range = [5,15]      # Low and high peak values for map blobs
spread_range = [5,12]    # Low and high spread distances for map blobs

delta_costs = [-1, 1];   # How many stds above and below for FM sampling
delta_costs2 = [-1];

nmethods = 4;
plot_timer = 10;

num_obstacles = 40       # Total number of obstacles
obstacle_size = 10       # Obstacle size

DATA_DIR = '/home/nick/Dropbox/work/FastMarching/data/'
VID_DIR = '/home/nick/Dropbox/work/FastMarching/explorer_vid/'
FIG_DIR = '/home/nick/Dropbox/work/FastMarching/fig_explore/'

my_blobs = [[30, 20, 16, 10], [60, 40,  6, 15], [10, 40, 32, 12], [60, 5, 15, 9],
            [25, 35,  8, 15], [70, 30, 12, 12], [80, 40, 16, 15], [5, 60, 30, 7],
            [30, 70, 32, 18], [90, 10, 20, 15]]

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
    
def check_row(mat, row):
    return np.equal(mat,row).all(1).any()
    ## Check if a row is equal to an existing row in a matrix
    #for ii in range(mat.shape[0]):
    #    this_row = True
    #    for jj in range(mat.shape[1]):
    #        this_row *= (mat[ii,jj] == row[jj])
    #    if this_row:
    #        return True
    #return False

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
    ax[0][1].set_title("Est. cost - Random sampling")
    ax[1][0].set_title("Est. cost - Max Variance sampling")
    ax[1][1].set_title("Est. cost - FMEx")
    #ax[1][2].set_title("Estimated cost - FM sampling2")
    return fig,ax

def plot_final_paths(ax, true_g, true_path, models):        
    axax = [ax[0][1], ax[1][0], ax[1][1]]#, ax[1][2]]
    graph_frame = []
    #if not ax[0][0].lines:
    tempframe, barlims = fm_plottools.draw_grid(ax[0][0], true_g, true_path, min_cost=mean_value-1)
    graph_frame.extend(tempframe)
    for i in range(len(models)):
        tempframe, barlims = fm_plottools.draw_grid(axax[i], models[i].GP_cost_graph, models[i].fbFM.path, max_cost=barlims[1],min_cost=barlims[0] )
        tempframe.append(axax[i].plot(models[i].X[:,0], models[i].X[:,1], 'rx')[0])
        graph_frame.extend(tempframe)
    return graph_frame

test_gridx = range(2, gridsize[0], 12);
test_gridy = range(2, gridsize[1], 12);

best_path_cost = np.zeros((1, NUM_STATRUNS), float)
true_path_cost = np.zeros((NUM_STATRUNS, nmethods, NUM_SAMPLES), float)
est_path_cost = np.zeros((NUM_STATRUNS, nmethods, NUM_SAMPLES), float)
est_path_var = np.zeros((NUM_STATRUNS, nmethods, NUM_SAMPLES), float)
sample_times = np.zeros((NUM_STATRUNS, NUM_SAMPLES), float)

if SAMPLING_VIDEOS:
    fig_s,ax_s = create_cost_plot()
if TIMED_PLOTS:
    fig_t,ax_t = create_cost_plot()
      
fig_v,ax_v = create_cost_plot()

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
fh.close()
print 'NOWSTR: {0}'.format(nowstr)

jj= 0

#for jj in range(NUM_STATRUNS):
while jj < NUM_STATRUNS:
    plt_tmr = plot_timer-1
    t0 = time.time()  
    cblobs = []
    sampling_frames=[]
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
        print "No map solution, recalculating, moving to next map".format(jj)
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
    random_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
    maxvar_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
    fm_sampling_explorer = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
    fm_sampling_explorer2 = bfm_explorer.fast_marching_explorer(gridsize, start_node, end_node, X, Y, mean_value, true_g.obstacles)
    random_sampling_explorer.search()
    maxvar_sampling_explorer.search()
    fm_sampling_explorer.search()
    fm_sampling_explorer2.search()
    
    ## UPDATES!
    
    for ii in range(NUM_SAMPLES):        
        # Random sampler:
        tx = random.choice(range(gridsize[0]))
        ty = random.choice(range(gridsize[1]))
        while ((tx,ty) in true_g.obstacles): # or (check_row(random_sampling_explorer.X, [tx,ty]))
            tx = random.choice(range(gridsize[0]))
            ty = random.choice(range(gridsize[1]))
        random_sampling_explorer.add_observation([tx,ty], sample_cost_fun(explore_cost_function, [tx,ty], cblobs))
        true_path_cost[jj,0,ii] = calc_true_path_cost(explore_cost_function, random_sampling_explorer.fbFM.path, cblobs)
        est_path_cost[jj,0,ii],est_path_var[jj,0,ii] = calc_est_path_cost(random_sampling_explorer.GP_model, mean_value, random_sampling_explorer.fbFM.path)
        
        # Max var and fm samplers
        max_var = 0
        fm_best_cost = -1
        fm_best_cost2 = -1
        
        ts = time.time()
        for whitmans_sampler in range(50):
            tx = random.choice(range(gridsize[0]))
            ty = random.choice(range(gridsize[1]))
            if True:
        #for tx in test_gridx:
        #    for ty in test_gridy:

                if  ((tx,ty) in true_g.obstacles):
                    continue
                # Max var
                # if not check_row(maxvar_sampling_explorer.X, [tx,ty]):
                t0 = time.time()
                temp_var = maxvar_sampling_explorer.varYfull[ty*gridsize[0]+tx]
                if temp_var > max_var:
                    max_var = temp_var
                    maxvar_bestX = [tx,ty]
            
                # if not check_row(fm_sampling_explorer.X, [tx,ty]):
                current_value = 0
                for td in delta_costs:
                    stdY = math.sqrt(fm_sampling_explorer.varYfull[ty*gridsize[0]+tx])
                    cost_update = poly_cost_obj.set_update(tx, ty, td*stdY)
                    current_value += fm_sampling_explorer.cost_update(cost_update)
                if fm_best_cost == -1 or (current_value < fm_best_cost):
                    fm_best_cost = current_value
                    fm_bestX = [tx,ty]

                # if not check_row(fm_sampling_explorer2.X, [tx,ty]):
                current_value = 0
                for td in delta_costs2:
                    stdY = math.sqrt(fm_sampling_explorer2.varYfull[ty*gridsize[0]+tx])
                    cost_update = poly_cost_obj.set_update(tx, ty, td*stdY)
                    current_value += fm_sampling_explorer2.cost_update_new(cost_update)
                if fm_best_cost2 == -1 or (current_value < fm_best_cost2):
                    fm_best_cost2 = current_value
                    fm_bestX2 = [tx,ty]

        #print "Sample selection time: {0:.4f}".format(time.time()-ts)
        # sample_times[jj,ii] = time.time()-ts
        maxvar_sampling_explorer.add_observation(maxvar_bestX, sample_cost_fun(explore_cost_function, maxvar_bestX, cblobs))
        true_path_cost[jj,1,ii] = calc_true_path_cost(explore_cost_function, maxvar_sampling_explorer.fbFM.path, cblobs)
        est_path_cost[jj,1,ii],est_path_var[jj,1,ii] = calc_est_path_cost(maxvar_sampling_explorer.GP_model, mean_value, maxvar_sampling_explorer.fbFM.path)
        
        # ts = time.time()
        fm_sampling_explorer.add_observation(fm_bestX, sample_cost_fun(explore_cost_function, fm_bestX, cblobs))
        # print('Update map time: {0}s'.format(time.time()-ts))
        true_path_cost[jj,2,ii] = calc_true_path_cost(explore_cost_function, fm_sampling_explorer.fbFM.path, cblobs)
        est_path_cost[jj,2,ii],est_path_var[jj,2,ii] = calc_est_path_cost(fm_sampling_explorer.GP_model, mean_value, fm_sampling_explorer.fbFM.path)
        
        fm_sampling_explorer2.add_observation(fm_bestX2, sample_cost_fun(explore_cost_function, fm_bestX2, cblobs))
        true_path_cost[jj,3,ii] = calc_true_path_cost(explore_cost_function, fm_sampling_explorer2.fbFM.path, cblobs)
        est_path_cost[jj,3,ii],est_path_var[jj,3,ii] = calc_est_path_cost(fm_sampling_explorer2.GP_model, mean_value, fm_sampling_explorer2.fbFM.path)
        
        if SAMPLING_VIDEOS:
            sampling_frames.append(plot_final_paths(ax_s, true_g, tFM.path, [random_sampling_explorer, maxvar_sampling_explorer, fm_sampling_explorer]))

        if TIMED_PLOTS and ii == plt_tmr:
            tf = plot_final_paths(ax_t, true_g, tFM.path, [random_sampling_explorer, maxvar_sampling_explorer, fm_sampling_explorer])
            fig_t.savefig(FIG_DIR+nowstr+'T{0}S{1}.pdf'.format(jj,ii+1), bbox_inches='tight')
            plt_tmr += plot_timer
            for item in tf:
                item.remove()
            
            
    
    if SAMPLING_VIDEOS:
        for exframe in range(5):
            sampling_frames.append(sampling_frames[-1])
        ani_sampling = animation.ArtistAnimation(fig_s, sampling_frames, interval=100, repeat_delay=0)
        ani_sampling.save('{0}{1}S{2:02d}.mp4'.format(VID_DIR, nowstr, jj), writer = 'avconv', fps=2, bitrate=1500)        
    best_path_cost[0][jj] = min([best_path_cost[0][jj], true_path_cost[jj,0,-1], true_path_cost[jj,1,-1], true_path_cost[jj,2,-1], true_path_cost[jj,3,-1]])
    video_frames.append(plot_final_paths(ax_v, true_g, tFM.path, [random_sampling_explorer, maxvar_sampling_explorer, fm_sampling_explorer]))
    if sampling_frames: 
        for item in sampling_frames[-1]:
            item.remove()
    print "STAT RUN {k}: Best: {cB:.2f}, Random:{cR:.2f}, MaxVar: {cV:.2f}, FM: {cF:.2f}, FM2: {cF2:.2f}. Total {t:.2f}s".format(k=jj, cB=best_path_cost[0][jj], cR=true_path_cost[jj,0,-1], cV=true_path_cost[jj,1,-1], cF=true_path_cost[jj,2,-1], cF2=true_path_cost[jj,3,-1], t=time.time()-t0)
    jj += 1

fh = open(DATA_DIR+nowstr+".p", "wb" )
pickle.dump(best_path_cost, fh)
pickle.dump(true_path_cost, fh)
pickle.dump(est_path_cost, fh)
pickle.dump(est_path_var, fh)
fh.close()


ani1 = animation.ArtistAnimation(fig_v, video_frames, interval=1000, repeat_delay=0)
ani1.save(VID_DIR+nowstr+'.mp4', writer = 'avconv', fps=1, bitrate=1500)

labels = ['Random', 'Max Variance', 'FMex', 'Fast March_{opt}']
fig1, fig2, fig3, fig4 = statrun_plots.make_plots(best_path_cost, true_path_cost, est_path_cost, labels)
fig1.savefig(FIG_DIR+nowstr+'C.pdf', bbox_inches='tight')
fig2.savefig(FIG_DIR+nowstr+'.pdf', bbox_inches='tight')
fig3.savefig(FIG_DIR+nowstr+'L.pdf', bbox_inches='tight')
fig3.savefig(FIG_DIR+nowstr+'E.pdf', bbox_inches='tight')
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