import numpy as np
import math
import random

# Basic cost function with sum of Gaussian-like blobs, and random generator

def blobby_cost_function(a, b, blobs, field_base_val=1.0):
    # blobs are defined by [xloc, yloc, spread, peakval]
    cost = field_base_val
    for i in range(np.shape(blobs)[0]):
        cost += blobs[i][3] * math.exp(-math.sqrt((a - blobs[i][0]) ** 2 + (b - blobs[i][1]) ** 2) / blobs[i][2])
    return cost

def gen_blobs(graph, n_blobs, spread_range=[1.0, 10.0], peak_range=[0, 1.0]):
    blobs = []
    for ii in range(n_blobs):
        xb = random.uniform(-10,graph.width+10)
        yb = random.uniform(-10,graph.height+10)
        rangeb = random.uniform(spread_range[0], spread_range[1])
        peakb = random.uniform(peak_range[0], peak_range[1])
        blobs.append([xb,yb, rangeb, peakb])
    return blobs

class mat_cost_function:
    # This is a faster way to compute the whole cost function on a grid so the cost lookup is just an array lookup
    def __init__(self, graph, cost_fun, *args, **kwargs):
        self.mat = np.zeros((graph.width, graph.height))
        self.left = graph.left
        self.bottom = graph.bottom
        for x in range(graph.width):
            for y in range(graph.height):
                self.mat[x, y] = cost_fun(self.left + x, self.bottom + y, *args, **kwargs)

    def calc_cost(self, a, b):
        return self.mat[a - self.left, b - self.bottom]