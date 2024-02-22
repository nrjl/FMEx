import matplotlib.pyplot as plt
import time
import math
from fm_tools import fast_marcher, fm_graphtools, fm_plottools
# from dijkstra_search import dijkstra_search, pull_path

def blob_cost_function(a, b):
    cost = 1 
    cost += 2*math.exp(-math.sqrt((a-25)**2 + (b-27)**2)/10)
    return cost

nowstr = time.strftime("%Y_%m_%d-%H_%M")

gridsize = [50, 50]

METHOD_NAMES = ['FM', 'E*', 'BiFM', 'FBFM', 'FBFM2']

g = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], blob_cost_function)
#g.obstacles = fm_plottools.generate_obstacles(g, 250, 10)
g.rebuild_neighbours()
start_node = (1,1)
end_node = (48,48) #'''
while end_node in g.obstacles:
    end_node = (end_node[0]-1, end_node[1])

models = [fast_marcher.FastMarcher(g), fast_marcher.FastMarcher(g), fast_marcher.BiFastMarcher(g), fast_marcher.FullBiFastMarcher(g), fast_marcher.FullBiFastMarcher(g)]
nmethods = len(METHOD_NAMES)

for model in models:
    model.set_start(start_node)
    model.set_goal(end_node)
    model.search()
    model.pull_path()

f0,a0 = plt.subplots(1,nmethods)
for i in range(nmethods):
    fm_plottools.draw_grid(a0[i], g, path=models[i].path)

poly_cost = fm_graphtools.polynomial_precompute_cost_modifier(g, 6, min_val=0.001)
cost_update =poly_cost.set_update(25, 25, 0.01)

ug = g.copy()
ug.add_delta_costs(cost_update)
uFM = fast_marcher.FastMarcher(ug)
uFM.set_start(start_node)
uFM.set_goal(end_node)
models[0] = uFM

models[0].search()
models[0].pull_path()
#models[1].update(cost_update)
#models[1].pull_path()
#models[2].update(cost_update)
#models[2].pull_path()

models[3].update(cost_update, recalc_path=True)
models[4].update_new3(cost_update, recalc_path=True)

f1,a1 = plt.subplots(1,nmethods)
fm_plottools.draw_grid(a1[0], ug, path=models[0].path)
fm_plottools.draw_grid(a1[1], ug, path=models[1].path)
fm_plottools.draw_grid(a1[2], ug, path=models[2].path)
fm_plottools.draw_grid(a1[3], ug, path=models[3].updated_path)
fm_plottools.draw_grid(a1[4], ug, path=models[4].updated_path)


            
#if i <= 2:
#    fm_plottools.draw_costmap(a1[i], g, models[i].cost_to_come)
#else:
#    fm_plottools.draw_costmap(a1[i], g, models[i].path_cost)
def fmex_val(x,y,z,model):
    model.update_new3(poly_cost.set_update(x,y,z))
    fmval = model.updated_min_path_cost-model.min_path_cost
    model.update_new3(poly_cost.set_update(x,y,-z))
    fmval2 = model.updated_min_path_cost-model.min_path_cost
    print("{0}: {1}, {2}: {3}, Total: {4}".format(z,fmval,-z,fmval2,fmval+fmval2))
    return fmval+fmval2



plt.show()