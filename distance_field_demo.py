import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fm_tools import fast_marcher, fm_graphtools, fm_plottools
from cost_functions import blobby_cost

# SEED_NUM = 1              # If you want a fixed map set this
# random.seed(SEED_NUM)           # Random seed

# Problem setup - generate a random field of Gaussian blobs as the cost field
plot_timer = 10
gridsize = [100, 100]    # Grid size

num_obstacles = 30       # Total number of obstacles
obstacle_size = 10       # Obstacle size


# Build new map (randomly generate Gaussian blobs and obstacles)
empty_graph = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1])

cblobs = blobby_cost.gen_blobs(empty_graph, 0)
cost_fun = blobby_cost.mat_cost_function(empty_graph, blobby_cost.blobby_cost_function, cblobs)
obstacles = fm_plottools.generate_obstacles(empty_graph, num_obstacles, obstacle_size)
graph = fm_graphtools.CostmapGridFixedObs(width=empty_graph.width, height=empty_graph.height,
                                          cost_fun=cost_fun.calc_cost, obstacles=obstacles)




# Plotting stuff
h_fig, h_ax = fm_plottools.init_fig(graph)

# Setup search. Obstacles are our start nodes
fm_object = fast_marcher.FastMarcher(graph)
fm_object.set_start_list(obstacles)
fm_object.set_plots([], h_ax)

# GO!
t0 = time.time()
fm_object.search()
t_searchFM = time.time()-t0
print "Search time: {0}s".format(t_searchFM)

# Plot animation because cool.
aniFM = animation.ArtistAnimation(h_fig, fm_object.make_video(), interval=10, repeat_delay=2000)
aniFM.save('vid/FM_{0}.ogg'.format(time.strftime("%Y_%m_%d-%H_%M")),
           writer='ffmpeg', fps=10, codec='libtheora', bitrate=8000)
plt.show()


