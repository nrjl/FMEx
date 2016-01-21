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

NUM_TESTS = 1000
random.seed(2)

search_time = np.zeros([3, NUM_TESTS], float)
search_nodes = np.zeros([3, NUM_TESTS], int)
downwind_nodes = np.zeros([3, NUM_TESTS], int)

def make_boxplot(data, lims, label, ax = 0):
    if ax == 0:
        ff = plt.figure()
        ax = ff.add_subplot(111)
    ax.boxplot(data)
    ax.set_ylim(lims[0], lims[1])
    ax.set_yscale('log')
    ax.grid(True, which='major', axis='y')
    ax.set_ylabel(label)
    ax.set_xticklabels(['FM', 'E*', 'BiFM'])
    return ax

print "Generating map..."
gridsize = [130, 100]

g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], fm_graphtools.blob_cost_function)
g.obstacles = fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 250, 10)
start_node = (1,1)
end_node = (127,97) #'''

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

## biFM search
bFM = fast_marcher.BiFastMarcher(g)
bFM.set_start(start_node)
bFM.set_goal(end_node)

print "Performing biFM search..."
t0 = time.time()
bFM.search()
t_searchbFM = time.time()-t0

bFM.find_corridor()
t_corridorbFM = time.time()-t0-t_searchbFM

print "Done. Search took {0}s, extracting corridor took {1}s".format(t_searchbFM, t_corridorbFM)

eFM = copy.deepcopy(FM)
ubFM = copy.deepcopy(bFM)

#bFM_reset(ubFM, bFM, g)
#ccost =fm_graphtools.square_cost_modifier(g, 25, 37, 88, 97, -0.991526711378)
#ubFM.update(ccost)

for ii in range(NUM_TESTS):
    xlo = random.randint(0, g.width-1)
    dx = random.randint(1, 20)
    ylo = random.randint(0, g.height-1)
    dy = random.randint(1, 20)
    modcost = 1.9*random.random()-1
    square_cost = fm_graphtools.square_cost_modifier(g, xlo, xlo+dx, ylo, ylo+dy, modcost)
    print "Random map update {0}, ({1},{2}) to ({3},{4}), delta = {5}".format(ii, xlo, ylo, xlo+dx, ylo+dy, modcost)
    print "Method | Time (s) | SNodes | DNodes |   Cost  |"
    
    ug = g.copy()
    ug.add_delta_costs(square_cost)
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
    eFM.update(square_cost, True)
    search_time[1,ii] = time.time()-t0
    search_nodes[1,ii] = eFM.search_nodes
    downwind_nodes[1,ii] = eFM.downwind_nodes
    print " E*FM  |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[1,ii], search_nodes[1,ii],downwind_nodes[1,ii],eFM.cost_to_come[eFM.end_node])

    # Update bFM
    fast_marcher.bFM_reset(ubFM, bFM, g)

    t0 = time.time()
    ubFM.update(square_cost)
    t_searchFMub = time.time()-t0
    search_time[2,ii] = time.time()-t0
    search_nodes[2,ii] = ubFM.search_nodes
    downwind_nodes[2,ii] = ubFM.downwind_nodes
    print " BiFM  |{0:10.4f}|{1:8.0f}|{2:8.0f}|{3:9.4f}|".format(search_time[2,ii], search_nodes[2,ii],downwind_nodes[2,ii],ubFM.best_cost)

trial = [search_time, search_nodes, downwind_nodes]
pickle.dump(trial, open( "../data/trial_new.p", "wb" ))

make_boxplot(np.transpose(search_time), [1e-5,15], 'Comp time (s)')

temp = [[search_time[jj,ii] for ii in range(NUM_TESTS) if search_nodes[2,ii] != 0] for jj in range(3)]
make_boxplot(temp, [1e-5,15], 'Comp time (s)')

temp2 = [[search_nodes[jj,ii]+downwind_nodes[jj,ii] for ii in range(NUM_TESTS)] for jj in range(3)]
make_boxplot(temp2, [1e-1,1e4], 'Total node expansions')

temp3 = [[search_nodes[jj,ii]+downwind_nodes[jj,ii] for ii in range(NUM_TESTS) if search_nodes[2,ii] != 0 ] for jj in range(3)]
make_boxplot(temp3, [1,1e4], 'Total node expansions')


ff = plt.figure()
ax1 = ff.add_subplot(121)
make_boxplot(np.transpose(search_time), [1e-6,15], 'Computational time (s)',ax1)
ax2 = ff.add_subplot(122)
make_boxplot(temp2, [1e-1,1e4], 'Total node expansions $(10^3)$',ax2)
ax2.set_yscale('linear')
ax2.set_ylim([-500, 15500])
ax2.set_yticklabels([str(aa) for aa in range(-2,15,2)])
ax2.yaxis.set_label_position('left')
ax2.yaxis.tick_right()
ff.set_size_inches([6,3])
ff.savefig('../fig/random_results.pdf', bbox_inches='tight')


plt.show()