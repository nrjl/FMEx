import fm_graphtools
import fm_plottools
import blobby_cost
import fast_marcher
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# SEED_NUM = 1              # If you want a fixed map set this
# random.seed(SEED_NUM)           # Random seed

# Problem setup - generate a random field of Gaussian blobs as the cost field
plot_timer = 10
gridsize = [100, 100]    # Grid size
field_base_val = 1.0           # Mean value of the field for GP
num_blobs = 30             # Number of blobs in each field
peak_range = [-3.0,8.0]    # Low and high peak values for map blobs
spread_range = [5,12]      # Low and high spread distances for map blobs

num_obstacles = 10       # Total number of obstacles
obstacle_size = 20       # Obstacle size


# Build new map (randomly generate Gaussian blobs and obstacles)
empty_graph = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1])

cblobs = blobby_cost.gen_blobs(empty_graph, num_blobs)
cost_fun = blobby_cost.mat_cost_function(empty_graph, blobby_cost.blobby_cost_function, cblobs)
obstacles = fm_plottools.generate_obstacles(empty_graph, num_obstacles, obstacle_size)
graph = fm_graphtools.CostmapGridFixedObs(width=empty_graph.width, height=empty_graph.height,
                                          cost_fun=cost_fun.calc_cost, obstacles=obstacles)

# Generate start and end nodes that aren't in obstacles
start_node = (3,3)
while start_node in graph.obstacles:
    start_node = (start_node[0]+1, start_node[1])
goal_node = (97, 97)
while goal_node in graph.obstacles:
    goal_node = (goal_node[0]-1, goal_node[1])

# Plotting stuff
h_fig, h_ax = fm_plottools.init_fig(graph)

# Setup search
fm_object = fast_marcher.FastMarcher(graph)
fm_object.set_start(start_node)
fm_object.set_goal(goal_node)
fm_object.set_plots([], h_ax)

# GO!
t0 = time.time()
fm_object.search()
t_searchFM = time.time()-t0
fm_object.pull_path()
t_pathFM = time.time()-t0-t_searchFM
print "Search time: {0}s, Pulling path time: {1}s".format(t_searchFM, t_pathFM)

# Plot animation because cool.
aniFM = animation.ArtistAnimation(h_fig, fm_object.make_video(), interval=10, repeat_delay=2000)
# aniFM.save('FM_{0}.ogg'.format(time.strftime("%Y_%m_%d-%H_%M")), writer='avconv', fps=2, codec='libtheora', bitrate=8000)
plt.show()


