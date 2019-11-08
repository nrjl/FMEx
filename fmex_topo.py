import numpy as np
import random
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
import pickle as pickle
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#import bfm_explorer
import GPy
import time
import bfm_explorer

topo_file = 'iceland_uk.p'
grid_file = 'iceland_uk_grid.p'
downsample_degree = 10
start_node = (17,123)#(100,100) #
end_node = (182,4)#(150,50) #
cost_scaler = 0.9

NUM_SAMPLES = 40

GP_mean = 0.36
sample_noise = 0.05
fmex_samples = 50

methods = ['maxvar','lcb','fmex']
ex_plot_index = [0,1,2]

plot_timer = 5
ex_plot_index = [0,1,2]

n_fmex_samples = 50

SEED_NUM = 2
random.seed(SEED_NUM)

lcb_weight = 1.0

DATA_DIR = '../data/'
VID_DIR = '../vid/'
FIG_DIR = '../fig/'
nowstr = time.strftime("%Y_%m_%d-%H_%M")
print "nowstr: {0}".format(nowstr)

def block_reduce(mat, degree):
    nx,ny = mat.shape
    newx,newy = (nx/degree, ny/degree)
    newmat = np.zeros((newx, newy), dtype=mat.dtype)
    for i in range(newx):
        for j in range(newy):
            newmat[i,j] = mat[i*degree:i*degree+degree,j*degree:j*degree+degree].mean()
    return newmat

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

def build_graph(grid_z):
    # Create cost map
    nx = grid_z.shape[0]
    ny = grid_z.shape[1]
    cost_z = -np.array(grid_z)
    cost_z /= (1.0/cost_scaler)*cost_z.max() # Want to be in range [1-cs, 1]
    cost_z += (1.0-cost_scaler)
    cost_fun = lambda x,y : cost_z[int(x),int(y)]
    cost_x = np.arange(nx) #grid_x[:,0].ravel()
    cost_y = np.arange(ny) #grid_y[0,:].ravel()
    spline_fit = interpolate.RectBivariateSpline(cost_x,cost_y,cost_z)
    obs = [(x,y) for x in range(nx) for y in range(ny) if grid_z[x,y] >= 0]
    outgraph = fm_graphtools.CostmapGridFixedObs(nx, ny, cost_fun, obstacles=obs)
    return outgraph,cost_fun,spline_fit

def sample_cost_fun(cost_fun, x):
    return cost_fun(x[0], x[1]) + random.gauss(0, sample_noise)
    
def gen_ax_labels(hl,lim):
    hl = (hl - hl[0])/(hl[-1]-hl[0])
    ll = lim[0]+hl*(lim[1]-lim[0])
    ll = ['{0:0.1f}'.format(x) for x in ll]
    return ll
    
def create_cost_plot(graph=None,labels=None,limits=None):
    fig, ax = plt.subplots(2, 2)
    if graph != None:
        for axn in ax.flat:
            axn.set_aspect('equal')
            axn.tick_params(labelbottom='on',labeltop='off')
            axn.set_xlim(-0.5, graph.width-.5)
            axn.set_ylim(-0.5, graph.height-0.5)
            if limits != None:
                ll = gen_ax_labels(np.array(axn.get_xticks()),limits[:2])
                axn.set_xticklabels(ll)
                axn.set_xlabel('Longitude (deg)')
                ll = gen_ax_labels(np.array(axn.get_yticks()),limits[2:])
                axn.set_yticklabels(ll)
                axn.set_ylabel('Latitude (deg)')
    else:
        for axn in ax.flat:
            axn.autoscale(tight=True)
 
    if labels != None:
        for aa,label in zip(ax.ravel(),labels):
            aa.set_title(label)    
    return fig,ax

def plot_final_paths(ax, true_g, true_path, models, true_costs=None, est_costs=None):        
    axax = ax.flat[1:]
    graph_frame = []
    #if not ax[0][0].lines:
    tempframe, barlims = fm_plottools.draw_grid(ax[0][0], true_g, true_path)
    graph_frame.extend(tempframe)
    for i in range(len(models)):
        tempframe, barlims = fm_plottools.draw_grid(axax[i], models[i].GP_cost_graph, path=models[i].fbFM.path, max_cost=barlims[1],min_cost=barlims[0] )
        tempframe.append(axax[i].plot(models[i].X[:,0], models[i].X[:,1], 'rx', mew=1.0, ms=8)[0])
        graph_frame.extend(tempframe)
    if true_costs is not None:
        for hax,cost in zip(ax.ravel(),true_costs):
            graph_frame.append(hax.text(10,10,'{0:0.2f}'.format(cost),fontsize='9.0',color='w',bbox=dict(facecolor='k', alpha=0.5)))
    if est_costs is not None:
        for ii,cost in enumerate(est_costs):
            graph_frame.append(axax[ii].text(200,117,'{0:0.2f}'.format(cost),fontsize='9.0',color='maroon',bbox=dict(facecolor='whitesmoke', alpha=0.8)))
    return graph_frame

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2)
    return y
    
def fmex_constructor(g, start_node, end_node, X, Y, sample_method, GP_l, GP_sv, GP_sn, GP_mean):
    fmex = bfm_explorer.fast_marching_explorer([g.width,g.height], start_node, end_node, X, Y, mean_value=GP_mean, obs=g.obstacles,
        GP_l=GP_l, GP_sv=GP_sv, GP_sn=GP_sn)
    fmex.set_sampler(sample_method)
    fmex.search()
    return fmex  


try:
    with open(DATA_DIR+grid_file, 'rb') as fp:
        grid_x = pickle.load(fp)
        grid_y = pickle.load(fp)
        grid_z = pickle.load(fp)

except IOError:
    print "Topography grid file {0} not found, creating {1}...".format(topo_file, grid_file)
    try:         
        with open(DATA_DIR+topo_file, 'rb') as fp:
            print "Loading pre-saved data file {0}.".format(topo_file)
            TOPO = pickle.load(fp)
        grid_x = TOPO['lons']
        grid_y = TOPO['lats']
        grid_z = TOPO['topo']
        if downsample_degree != 1:
            # Downsample:
            grid_x = block_reduce(grid_x, downsample_degree)
            grid_y = block_reduce(grid_y, downsample_degree)
            grid_z = block_reduce(grid_z, downsample_degree)
        with open(DATA_DIR+grid_file, 'wb') as fp2:
            pickle.dump(grid_x, fp2)
            pickle.dump(grid_y, fp2)
            pickle.dump(grid_z, fp2)
                
    except IOError:
        print "Topography file {0} not found, create using plot_world_topo".format(topo_file)
        raise 

minlon = grid_x[0,0]
maxlon = grid_x[-1,0]
minlat = grid_y[0,0]
maxlat = grid_y[0,-1]
   
# Create map
m = Basemap(projection='mill', llcrnrlat=minlat,urcrnrlat=maxlat,llcrnrlon=minlon, urcrnrlon=maxlon,resolution='i')
x,y = m(grid_x,grid_y)

hf,ha = plt.subplots()
extent = [minlon,maxlon,minlat,maxlat]
cmap = plt.cm.viridis
cmap.set_bad(color='black')
hmat = ha.imshow((np.ma.masked_where(grid_z > 0, grid_z)).transpose(),
    origin='lower',interpolation='none', cmap=cmap, extent=extent)
ha.axis('equal')
ha.axis(extent)
ha.set_title('SMRT30 Bathymetry')
ha.set_xlabel('Latitude (deg)')
ha.set_ylabel('Longitude (deg)')
cbar = plt.colorbar(hmat,orientation='horizontal')
cbar.ax.set_xlabel('meters')
hf.show()

labels = ['Max Var ({0})'.format(grid_z.shape[0]*grid_z.shape[1]), 
        'LCB ({0})'.format(grid_z.shape[0]*grid_z.shape[1]),
        'FMEx $3\sigma$ ({0})'.format(n_fmex_samples)]
        
# Plot cost map
hf2,ha2 = plt.subplots()
true_g,cost_fun,spline_cost = build_graph(grid_z)

tFM = fast_marcher.FullBiFastMarcher(true_g)
tFM.set_start(start_node)
tFM.set_goal(end_node)
tFM.search()
tFM.pull_path()
hart,lims = fm_plottools.draw_grid(ha2, true_g, tFM.path)
cbar = plt.colorbar(hart[0])
best_path_cost = calc_true_path_cost(spline_cost, tFM.path)
# Calculate GP hyperparameters

# Sample field
gp_X = []
while len(gp_X) < 500:
    tx = random.randrange(grid_z.shape[0])
    ty = random.randrange(grid_z.shape[1])
    if ((tx,ty) not in true_g.obstacles):
        gp_X.append([tx,ty])
gp_X = np.array(gp_X)
gp_Y = np.zeros((gp_X.shape[0],1))
for ii,x in enumerate(gp_X):
    gp_Y[ii] = sample_cost_fun(cost_fun, x)
GP_model = GPy.models.GPRegression(gp_X,gp_Y,GPy.kern.RBF(2))
GP_model.optimize()
GP_l = GP_model.rbf.lengthscale[0]
GP_sv = GP_model.rbf.variance[0]
GP_sn = GP_model.Gaussian_noise.variance[0]
print "GP hypers: l={0:0.2f}, s_f={1:0.2f}, s_n={2:0.2f}".format(GP_l, np.sqrt(GP_sv), np.sqrt(GP_sn)) 

GP_model.plot()

poly_cost_obj = fm_graphtools.polynomial_precompute_cost_modifier(true_g, int(2.0*GP_l), min_val=0.001)

# Create initial GP samples
# X = np.array([[start_node[0], start_node[1]]])
X = np.array([[-50, -50]])
Y = np.zeros((X.shape[0], 1))
for ii in range(X.shape[0]):
    Y[ii] = sample_cost_fun(cost_fun, X[0])

# Create BiFM explorer objects for each sampling strategy
t0 = time.time()
gridsize = (grid_z.shape[0],grid_z.shape[1])
explorers = [fmex_constructor(true_g, start_node, end_node, X, Y, method, GP_l=0.5*GP_l, GP_sv=0.25, GP_sn=GP_sn, GP_mean=GP_mean) for method in methods]
nmethods = len(methods)

llab = ["True cost field"]
llab.extend([labels[i] for i in ex_plot_index])
fig_s,ax_s = create_cost_plot(true_g, llab, extent)
fig_t,ax_t = create_cost_plot(true_g, llab, extent)
fig_t.set_size_inches(np.array([ 9.8 ,  6]))

## UPDATES!
fm_3sigma = [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0]
fm_3sigma_w = [bfm_explorer.normpdf(x, 0.0, 1.0) for x in fm_3sigma]

true_path_cost = np.zeros((nmethods, NUM_SAMPLES), dtype='float')
est_path_cost = np.zeros((nmethods, NUM_SAMPLES), dtype='float')
est_path_var = np.zeros((nmethods, NUM_SAMPLES), dtype='float')
current_true_costs = np.zeros(len(ex_plot_index)+1)
current_est_costs = np.zeros(len(ex_plot_index))
sampling_frames=[]
current_true_costs[0] = best_path_cost

for ii in range(NUM_SAMPLES):        
    
    ts = time.time(); 
    bestX = np.zeros((nmethods,2))

    ## Run samplers:
    bestX[0,:] = explorers[0].run_counted_sampler();
    bestX[1,:] = explorers[1].run_counted_sampler(cweight=lcb_weight);
    bestX[2,:] = explorers[2].run_counted_sampler(n_fmex_samples, 
            cost_obj=poly_cost_obj, delta_costs=fm_3sigma, weights=fm_3sigma_w)
    if ii == 0:
        bestX[2,:] = [70,80]
        bestX[1,:] = [100,20]
    elif ii == 1:
        bestX[2,:] = [90,115]
        
    for dex, explorer in enumerate(explorers):
        explorer.add_observation([bestX[dex,:]], [[sample_cost_fun(cost_fun, bestX[dex,:])]])
        true_path_cost[dex,ii] = calc_true_path_cost(spline_cost, explorer.fbFM.path)
        est_path_cost[dex,ii],est_path_var[dex,ii] = calc_est_path_cost(explorer.GP_model, GP_mean, explorer.fbFM.path)

    #print('Update map time: {0}s'.format(time.time()-ts))
    
    for ei in range(len(ex_plot_index)):
        current_true_costs[ei+1] = true_path_cost[ex_plot_index[ei],ii]
        current_est_costs[ei] = est_path_cost[ex_plot_index[ei],ii]
            
    sampling_frames.append(plot_final_paths(ax_s, true_g, tFM.path, 
        [explorers[nn] for nn in ex_plot_index],
        true_costs=current_true_costs, est_costs=current_est_costs))

    if (ii+2) % plot_timer == 0:
        tf = plot_final_paths(ax_t, true_g, tFM.path, 
            [explorers[nn] for nn in ex_plot_index],
            true_costs=current_true_costs, est_costs=current_est_costs)
        fig_t.savefig(FIG_DIR+topo_file+'{0}S{1}.pdf'.format(nowstr,ii+2), bbox_inches='tight')
        for item in tf:
            item.remove()
    print "Sample {0} selected, t={1}s, costs={2}".format(ii, time.time()-ts,current_true_costs)


for exframe in range(5):
    sampling_frames.append(sampling_frames[-1])
ani_sampling = animation.ArtistAnimation(fig_s, sampling_frames, interval=100, repeat_delay=0)
ani_sampling.save('{0}{1}{2}S{3}.ogg'.format(VID_DIR, topo_file, nowstr,ii), writer = 'avconv', fps=2, bitrate=8000, codec='libtheora')   
print "Total  {t:.2f}s".format(t=time.time()-t0)


plt.show()
 
# plt.savefig('../fig/topo.png', bbox_inches='tight')