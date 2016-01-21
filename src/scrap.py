import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import fast_marcher
import pickle

import fm_graphtools
import fm_plottools

gridsize = [100, 100]
obs = fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 100, 10)
G1 = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], fm_graphtools.unit_cost_function, obs)
G2 = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], fm_graphtools.unit_cost_function, obs)
t0 = time.time()
for x in range(gridsize[0]):
    for y in range(gridsize[1]):
        temp = G2.neighbours((x,y))
t2 = time.time()-t0
for x in range(gridsize[0]):
    for y in range(gridsize[1]):
        temp = G1.neighbours((x,y))
t1 = time.time()-t0-t2
print "CostmapGrid: {0}s, FixedObs: {1}s".format(t1, t2)


def square_cost_modifier(graph, xlo, xhi, ylo, yhi, delta):
    cost_dict={}
    for x in range(xlo, min(graph.width, xhi+1)):
        for y in range(ylo, min(graph.height, yhi+1)):
            if (x,y) not in graph.obstacles:
                cost_dict[(x,y)] = delta
    return cost_dict

gridsize = [130, 100]

random.seed(3)
g = fm_graphtools.CostmapGrid(gridsize[0], gridsize[1], fm_graphtools.blob_cost_function)

g.obstacles = fm_plottools.generate_obstacles(gridsize[0], gridsize[1], 250, 10)
start_node = (1,1)
end_node = (127,97) #'''

ug = g.copy()
eFM = copy.copy(FM)
eFM.set_graph(ug)
eFM.downwind_nodes = 0
t0 = time.time()
eFM.update(square_cost, True)











        ## BROKEN!!!!!!!!!!!! ##################################################
        # New cost should be added as a dictionary, with elements  [(node) : delta_cost]
        midnode = (-1,-1)
        
        # If it was a cost increase:
        #   - If the minimum path cost in the whole region is greater than the 
        #       exisiting best path cost, then quit? (invalidate the original best path!)
        #   - If the best path passes through the region, then we have to update
        #   - Need the kill list (downwind of cost increases)
        # If it was a cost decrease:
        #   - No need for a kill list
        #   - We have to update (but can still terminate early using the minimum
        #       cost to goal and current cost)
        
        # Strip zero cost nodes and obstacle nodes
        new_cost = {node:new_cost[node] for node in new_cost if ((new_cost[node] != 0) and (node in self.path_cost))}
        self.graph.clear_delta_costs()
        self.graph.add_delta_costs(new_cost)
        
        # Find boundary points closest to start and goal
        min_cts = self.min_path_cost
        min_ctg = self.min_path_cost
        
        # Find the lowest-cost node on both searches
        cost_to_come = copy.copy(self.FastMarcherSG.cost_to_come)
        for node in new_cost:
            for pnode in self.FastMarcherSG.parent_list[node]:
                if cost_to_come[pnode] < min_cts:
                    start_start = pnode
                    min_cts = cost_to_come[pnode]
            for pnode in self.FastMarcherGS.parent_list[node]:
                if self.FastMarcherGS.cost_to_come[pnode] < min_ctg:
                    start_goal = pnode
                    min_ctg = self.FastMarcherGS.cost_to_come[pnode]                
    
        if min_cts+min_ctg > self.min_path_cost:
            self.updated_min_path_cost = self.min_path_cost
            return

            # Check if the update was a cost increase or decrease
        if new_cost.values > 0:
            cost_increase = True
        else:
            cost_increase = False
                
        # Create frontier with just edge nodes closest to start and goal        
        frontier = fm_graphtools.PriorityQueue([])
        frontier.clear()
        frontier.push(start_start, min_cts)
        finished = False
        nodes_popped = 0

        while (self.frontier.count() > 0) and finished == False:
            try:
                c_priority, c_node = self.frontier.pop()
                nodes_popped+=1
                if self.FastMarcherGS.cost_to_come[c_node] <= min_ctg:
                    return c_priority+self.FastMarcherGS.cost_to_come[c_node]
            except ValueError:
                continue
            except KeyError:
                break
            u_A = c_priority
            cost_to_come[c_node] = u_A
            for n_node, tau_k in self.graph.neighbours(c_node):
                u_B = u_A + tau_k + 1.0
                adjacency = (n_node[0]-c_node[0], n_node[1]-c_node[1])
                for adjacent_node in self.FastMarcherSG.adjacency_list[adjacency]:
                    B_node = (n_node[0]+adjacent_node[0], n_node[1]+adjacent_node[1])
                    if B_node in cost_to_come and cost_to_come[B_node] < u_B:
                        u_B = cost_to_come[B_node]
                        
                if tau_k > abs(u_A - u_B):
                    c_cost = 0.5*(u_A + u_B + math.sqrt(2*tau_k**2 - (u_A - u_B)**2))
                else:
                    if u_A <= u_B:
                        c_cost = u_A + tau_k
                    else:
                        c_cost = u_B + tau_k
                
                if n_node not in cost_to_come:
                    frontier.push(n_node, c_cost)
                elif n_node in interface:
                    if c_cost + self.FastMarcherGS.cost_to_come[n_node] < self.updated_min_path_cost:
                        self.updated_min_path_cost = c_cost + self.FastMarcherGS.cost_to_come[n_node]
                        midnode = c_node
                        #print "Better path found!"                
                    
            if self.image_frames != 0 and u_A > self.plot_cost :            
                self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, cost_to_come))
                self.plot_cost = u_A + self.delta_plot
            
                
        if self.image_frames != 0:
            self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, self.cost_to_come))
        # print "biFM search: nodes popped: {0}".format(nodes_popped)
        self.search_nodes=nodes_popped    
        # Append final frame
        if self.image_frames != 0:
            self.image_frames.append(fm_plottools.draw_costmap(self.axes, self.graph, cost_to_come))
        # print "fbFM update: nodes popped: {0}".format(nodes_popped)
        self.search_nodes = nodes_popped
        
        if recalc_path:
            if midnode != (-1,-1):
                tempctc = copy.copy(self.FastMarcherSG.cost_to_come)
                self.FastMarcherSG.cost_to_come = cost_to_come
                path1 = self.FastMarcherSG.path_source_to_point(self.start_node, midnode)
                path2 = self.FastMarcherGS.path_source_to_point(self.end_node, midnode)
                path2.remove(midnode)
            
                path2.reverse()
                path1.extend(path2)
                self.updated_path = path1
                self.FastMarcherSG.cost_to_come = tempctc
            else:
                self.updated_path = copy.copy(self.path)

        return 