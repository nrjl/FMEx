import numpy as np
import random
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
import pickle as pickle
from fm_tools import fast_marcher, fm_graphtools, fm_plottools
# import bfm_explorer
import GPy
import time
import bfm_explorer

topo_file = 'iceland_uk.p'
grid_file = 'iceland_uk_grid.p'
downsample_degree = 10
start_node = (17, 123)  # (100,100) #
end_node = (182, 4)  # (150,50) #
cost_scaler = 0.9

GP_mean = 0.36
sample_noise = 0.05
fmex_samples = 50

DATA_DIR = '../data/'
VID_DIR = '../vid/'
FIG_DIR = '../../../pubs/RAL17_FM/presentation/'
nowstr = time.strftime("%Y_%m_%d-%H_%M")

def calc_true_path_cost(cost_fun, path, *args):
    true_cost = 0
    for i in range(len(path) - 1):
        d = np.sqrt((path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2)
        true_cost += (cost_fun(path[i][0], path[i][1], *args) +
                      cost_fun(path[i + 1][0], path[i + 1][1], *args)) / 2 * d
    return true_cost


def calc_est_path_cost(gp_model, mean_val, path):
    true_cost, var_cost = gp_model.predict(np.asarray(path))
    true_cost += mean_val
    sum_cost = 0
    sum_var = 0
    for i in range(len(path) - 1):
        d = np.sqrt((path[i][0] - path[i + 1][0]) ** 2 + (path[i][1] - path[i + 1][1]) ** 2)
        sum_cost += (true_cost[i] + true_cost[i + 1]) / 2 * d
        sum_var += (var_cost[i] + var_cost[i + 1]) / 2 * d
    return sum_cost, sum_var


def build_graph(grid_z):
    # Create cost map
    nx = grid_z.shape[0]
    ny = grid_z.shape[1]
    cost_z = -np.array(grid_z)
    cost_z /= (1.0 / cost_scaler) * cost_z.max()  # Want to be in range [1-cs, 1]
    cost_z += (1.0 - cost_scaler)
    cost_fun = lambda x, y: cost_z[int(x), int(y)]
    cost_x = np.arange(nx)  # grid_x[:,0].ravel()
    cost_y = np.arange(ny)  # grid_y[0,:].ravel()
    spline_fit = interpolate.RectBivariateSpline(cost_x, cost_y, cost_z)
    obs = [(x, y) for x in range(nx) for y in range(ny) if grid_z[x, y] >= 0]
    outgraph = fm_graphtools.CostmapGridFixedObs(nx, ny, cost_fun, obstacles=obs)
    return outgraph, cost_fun, spline_fit


def sample_cost_fun(cost_fun, x):
    return cost_fun(x[0], x[1]) + random.gauss(0, sample_noise)


def gen_ax_labels(hl, lim):
    hl = (hl - hl[0]) / (hl[-1] - hl[0])
    ll = lim[0] + hl * (lim[1] - lim[0])
    ll = ['{0:0.1f}'.format(x) for x in ll]
    return ll


def create_cost_plot(graph=None, title=None, limits=None):
    fig, axn = plt.subplots()
    if graph != None:
        axn.set_aspect('equal')
        axn.tick_params(labelbottom='on', labeltop='off')
        axn.set_xlim(-0.5, graph.width - .5)
        axn.set_ylim(-0.5, graph.height - 0.5)
        if limits != None:
            ll = gen_ax_labels(np.array(axn.get_xticks()), limits[:2])
            axn.set_xticklabels(ll)
            axn.set_xlabel('Longitude (deg)')
            ll = gen_ax_labels(np.array(axn.get_yticks()), limits[2:])
            axn.set_yticklabels(ll)
            axn.set_ylabel('Latitude (deg)')
    else:
        axn.autoscale(tight=True)

    if title != None:
        axn.set_title(title)
    return fig, axn

with open(DATA_DIR + grid_file, 'rb') as fp:
    grid_x = pickle.load(fp)
    grid_y = pickle.load(fp)
    grid_z = pickle.load(fp)

minlon = grid_x[0, 0]
maxlon = grid_x[-1, 0]
minlat = grid_y[0, 0]
maxlat = grid_y[0, -1]

# Create map
m = Basemap(projection='mill', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon, urcrnrlon=maxlon, resolution='i')
x, y = m(grid_x, grid_y)
extent = [minlon,maxlon,minlat,maxlat]
cmap = plt.cm.viridis
cmap.set_bad(color='black')
true_g, cost_fun, spline_cost = build_graph(grid_z)

tFM = fast_marcher.FullBiFastMarcher(true_g)
tFM.set_start(start_node)
tFM.set_goal(end_node)
tFM.search()
tFM.pull_path()
best_path_cost = calc_true_path_cost(spline_cost, tFM.path)[0][0]

# Create initial GP samples
obs_only = fm_graphtools.CostmapGridFixedObs(true_g.width, true_g.height, obstacles=true_g.obstacles)


# Plot with only obstacles and end points
fig_s, ax_s = create_cost_plot(true_g, title='Iceland-UK', limits=extent)
fm_plottools.draw_grid(ax_s, obs_only, min_cost = 0.8, max_cost=2.0)
fm_plottools.plot_end_points(ax_s, *list(zip(*tFM.path)))

# Plot with true cost map, no path
fig_t, ax_t = create_cost_plot(true_g, title='Iceland-UK SMRT30 Bathymetry', limits=extent)
tempframe, barlims = fm_plottools.draw_grid(ax_t, true_g)
fm_plottools.plot_end_points(ax_t, *list(zip(*tFM.path)))

# Plot true cost map with path
fig_p, ax_p = create_cost_plot(true_g, title='Iceland-UK SMRT30 Bathymetry', limits=extent)
tempframe, barlims = fm_plottools.draw_grid(ax_p, true_g, tFM.path)
ax_p.text(10, 10, '{0:0.2f}'.format(best_path_cost), fontsize='9.0', color='w', bbox=dict(facecolor='k', alpha=0.5))

# Plot estimate
fig_e, ax_e = create_cost_plot(true_g, title='Iceland-UK Estimate', limits=extent)
X = np.array([[40,45], [60,76],  [80,85], [32,113], [83,114],  [103,120], [130,127], [114,109], [127,86], [159,79],
              [197,94], [215,95], [163,50], [160,21], [189,36], [220,56], [194,65], [196,79], [210,82], [151,90],
              [176,73], [141,95]])
GP_l, GP_sv, GP_sn = 15.0, 0.35, 0.07

Y = np.zeros((X.shape[0], 1))
for ii in range(X.shape[0]):
    Y[ii] = sample_cost_fun(cost_fun, X[ii])

fmex = bfm_explorer.fast_marching_explorer([true_g.width, true_g.height], start_node, end_node, X, Y, mean_value=GP_mean,
                                           obs=true_g.obstacles, GP_l=GP_l, GP_sv=GP_sv, GP_sn=GP_sn)
fmex.search()
e_tpathcost = calc_true_path_cost(spline_cost, fmex.fbFM.path)[0][0]
e_epathcost = calc_est_path_cost(fmex.GP_model, GP_mean, fmex.fbFM.path)[0][0]

fm_plottools.draw_grid(ax_e, fmex.GP_cost_graph, path=fmex.fbFM.path, max_cost=barlims[1],min_cost=barlims[0] )
ax_e.plot(X[:, 0], X[:, 1], 'rx', mew=1.0, ms=8)
ax_e.text(10, 10, '{0:0.2f}'.format(e_tpathcost), fontsize='9.0', color='w', bbox=dict(facecolor='k', alpha=0.5))
ax_e.text(200, 117, '{0:0.2f}'.format(e_epathcost), fontsize='9.0', color='maroon', bbox=dict(facecolor='whitesmoke', alpha=0.8))

for f, let in zip([fig_s, fig_p, fig_t, fig_e], ['s','p','t','e']):
    f.set_size_inches(np.array([6, 4]))
    f.savefig(FIG_DIR+let+'.png', bbox_inches='tight')

plt.show()

# plt.savefig('../fig/topo.png', bbox_inches='tight')