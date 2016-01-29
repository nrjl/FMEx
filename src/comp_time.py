import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
import fast_marcher
import pickle

import fm_graphtools
import fm_plottools
# from dijkstra_search import dijkstra_search, pull_path

nowstr = time.strftime("%Y_%m_%d-%H_%M")

NUM_TESTS = 1000
random.seed(2)

METHOD_NAMES = ['FM', 'E*', 'BiFM', 'FBFM', 'BiFM']
nmethods = len(METHOD_NAMES)
search_time = np.zeros([nmethods, NUM_TESTS], float)
search_nodes = np.zeros([nmethods, NUM_TESTS], int)
downwind_nodes = np.zeros([nmethods, NUM_TESTS], int)

def make_boxplot(data, lims, ylabel, ax = None, title=None, selector=None):
    data = np.array(data)
    if ax == None:
        ff = plt.figure()
        ax = ff.add_subplot(111)
    if selector == None:
        selector = range(data.shape[1])
    data = data[:,selector]
    dlabels = [METHOD_NAMES[i] for i in selector]
    ax.boxplot(data)
    ax.set_ylim(lims[0], lims[1])
    ax.set_yscale('log')
    ax.grid(True, which='major', axis='y')
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(dlabels)
    if title != None:
        ax.set_title(title)
    return ax

print "Generating map..."
gridsize = [100, 100]

g = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], fm_graphtools.blob_cost_function)
g.obstacles = fm_plottools.generate_obstacles(g, 20, 20)
g.rebuild_neighbours()
start_node = (1,1)
end_node = (97,97) #'''
while end_node in g.obstacles:
    end_node = (end_node[0]-1, end_node[1])
    
# FM search
print "Performing FM search..."
FM = fast_marcher.FastMarcher(g)
FM.set_start(start_node)
FM.set_goal(end_node)

t0 = time.time()
FM.search()
t_searchFM = time.time()-t0

FM.find_corridor()
t_corridorFM = time.time()-t0-t_searchFM
print "Done. Search took {0}s, extracting corridor took {1}s".format(t_searchFM, t_corridorFM)

FM.pull_path()
f0,a0 = fm_plottools.init_fig()
fm_plottools.draw_grid(a0,g,path=FM.path)

print "Performing biFM search..."
bFM = fast_marcher.BiFastMarcher(g)
bFM.set_start(start_node)
bFM.set_goal(end_node)

t0 = time.time()
bFM.search()
t_searchbFM = time.time()-t0

bFM.find_corridor()
t_corridorbFM = time.time()-t0-t_searchbFM
print "Done. Search took {0}s, extracting corridor took {1}s".format(t_searchbFM, t_corridorbFM)

print "Performing full BiFM search..."
fbFM = fast_marcher.FullBiFastMarcher(g)
fbFM.set_start(start_node)
fbFM.set_goal(end_node)

t0 = time.time()
fbFM.search()
t_searchfbFM = time.time()-t0
print "Done. Search took {0}s".format(t_searchfbFM)



eFM = copy.deepcopy(FM)
ubFM = copy.deepcopy(bFM)

#bFM_reset(ubFM, bFM, g)
#ccost =fm_graphtools.square_cost_modifier(g, 25, 37, 88, 97, -0.991526711378)
#ubFM.update(ccost)

poly_cost = fm_graphtools.polynomial_precompute_cost_modifier(g, 12, min_val=0.001)

for ii in range(NUM_TESTS):
    xlo = random.randint(5, g.width-5)
    ylo = random.randint(5, g.height-5)
    modcost = 1.9*random.random()-1
    #dx = random.randint(1, 20)
    #dy = random.randint(1, 20)
    #cost_update = fm_graphtools.square_cost_modifier(g, xlo, xlo+dx, ylo, ylo+dy, modcost)
    #print "Square map update {0}, ({1},{2}) to ({3},{4}), delta = {5}".format(ii, xlo, ylo, xlo+dx, ylo+dy, modcost)
    cost_update = poly_cost.set_update(xlo,ylo,modcost)
    print "Poly map update {0}, ({1},{2}), delta = {3}".format(ii, xlo, ylo, modcost)
    print "Method | Time (s) | SNodes | DNodes |   Cost  |"
    
    ug = g.copy()
    ug.add_delta_costs(cost_update)
    uFM = fast_marcher.FastMarcher(ug)
    uFM.set_start(start_node)
    uFM.set_goal(end_node)

    t0 = time.time()
    uFM.search()
    search_time[0,ii] = time.time()-t0
    search_nodes[0,ii] = uFM.search_nodes
    print "  FM   |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[0,ii], search_nodes[0,ii],0,uFM.cost_to_come[uFM.end_node])

    # E* Update test
    fast_marcher.FM_reset(eFM, FM, g)
        
    t0 = time.time()
    eFM.update(cost_update, True)
    search_time[1,ii] = time.time()-t0
    search_nodes[1,ii] = eFM.search_nodes
    downwind_nodes[1,ii] = eFM.downwind_nodes
    print " E*FM  |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[1,ii], search_nodes[1,ii],downwind_nodes[1,ii],eFM.cost_to_come[eFM.end_node])

    # Update bFM
    fast_marcher.bFM_reset(ubFM, bFM, g)

    t0 = time.time()
    ubFM.update(cost_update)
    t_searchFMub = time.time()-t0
    search_time[2,ii] = time.time()-t0
    search_nodes[2,ii] = ubFM.search_nodes
    downwind_nodes[2,ii] = ubFM.downwind_nodes
    print " BiFM  |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[2,ii], search_nodes[2,ii],downwind_nodes[2,ii],ubFM.best_cost)
    
    # Update fbFM
    t0 = time.time()
    fbFM.update(cost_update)
    t_searchFMufb = time.time()-t0
    search_time[3,ii] = time.time()-t0
    search_nodes[3,ii] = fbFM.search_nodes
    downwind_nodes[3,ii] = fbFM.downwind_nodes
    print " FBFM  |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[3,ii], search_nodes[3,ii],downwind_nodes[3,ii],fbFM.updated_min_path_cost)

    # Update fbFM2
    t0 = time.time()
    fbFM.update_new3(cost_update)
    t_searchFMufb = time.time()-t0
    search_time[4,ii] = time.time()-t0
    search_nodes[4,ii] = fbFM.search_nodes
    downwind_nodes[4,ii] = fbFM.downwind_nodes
    print " FBFM3 |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[4,ii], search_nodes[4,ii],downwind_nodes[4,ii],fbFM.updated_min_path_cost)
    

trial = [search_time, search_nodes, downwind_nodes]
with open( "../data/update_times_{0}.p".format(nowstr), "wb" ) as fh:
    pickle.dump(trial, fh)

make_boxplot(np.transpose(search_time), [1e-6,1], 'Comp time (s)', title='Total search time')

temp = np.transpose([[search_time[jj,ii] for ii in range(NUM_TESTS) if search_nodes[2,ii] != 0] for jj in range(nmethods)])
make_boxplot(temp, [1e-5,15], 'Comp time (s)', title='Search time for non-null updates')

temp2 = np.transpose([[search_nodes[jj,ii]+downwind_nodes[jj,ii] for ii in range(NUM_TESTS)] for jj in range(nmethods)])
make_boxplot(temp2, [1e-1,1e4], 'Total node expansions', title='Node expansions')

temp3 = np.transpose([[search_nodes[jj,ii]+downwind_nodes[jj,ii] for ii in range(NUM_TESTS) if search_nodes[2,ii] != 0 ] for jj in range(nmethods)])
make_boxplot(temp3, [1,1e4], 'Total node expansions', title='Node expansions for non-null updates')


selector = [0,1,4]
ff = plt.figure(0); ff.clf()
ax1 = ff.add_subplot(121)
make_boxplot(np.transpose(search_time), [1e-5,1], 'Computational time (s)',ax=ax1, selector=selector)
ax2 = ff.add_subplot(122)
make_boxplot(temp2, [1e-1,1e4], 'Total node expansions $(10^3)$',ax=ax2, selector=selector)
ax2.set_yscale('linear')
ax2.set_ylim([-500, 15500])
ax2.set_yticklabels([str(aa) for aa in range(-2,15,2)])
ax2.yaxis.set_label_position('left')
ax2.yaxis.tick_right()
ff.set_size_inches([6,3])
ff.savefig('../fig/random_results{0}.pdf'.format(nowstr), bbox_inches='tight')
plt.show()