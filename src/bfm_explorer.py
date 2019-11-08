import numpy as np
import GPy
import fast_marcher
import fm_graphtools
import random
import math

def zero_fun(X):
    return 0,0

def normpdf(x, mu, sigma):
    u = (x-mu)/abs(sigma)
    y = (1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2)
    return y
    
class mat_cost_function:
    def __init__(self, graph, cost_fun=zero_fun, *args, **kwargs):
        self.mat = np.zeros((graph.width, graph.height))
        self.left = graph.left
        self.bottom = graph.bottom
        for x in range(graph.width):
            for y in range(graph.height):
                self.mat[x,y] = cost_fun(self.left+x, self.bottom+y, *args, **kwargs)
    
    def calc_cost(self,a,b):
        return self.mat[a-self.left,b-self.bottom]
        
class mat_cost_function_GP:
    def __init__(self, graph, cost_fun=zero_fun, mean_value=0, *args, **kwargs):
        lx = np.arange(graph.left, graph.right, dtype='int')
        ly = np.arange(graph.bottom, graph.top, dtype='int')
        X_star = np.array([[x,y] for x in lx for y in ly])
        Y_star,Y_var = cost_fun(X_star,*args, **kwargs)
        Y_star += mean_value
        self.cost_dict = {(X_star[k,0],X_star[k,1]):Y_star[k,0] for k in range(Y_star.shape[0])}
        self.var_dict  = {(X_star[k,0],X_star[k,1]):Y_var[k,0]  for k in range(Y_var.shape[0])}
    
    def calc_cost(self,a,b):
        return self.cost_dict[(a,b)]
        
    def calc_var(self,a,b):
        return self.var_dict[(a,b)]

def GPdepth_cost_function(X, GPm, max_depth=1.0e3, mean_depth=0.0):
    # Cost function shold be strictly positive (depth < max_depth)
    mean,var = GPm.predict(X)
    mean = max_depth-(mean+mean_depth)
    mean[mean < 0.1] = 0.1
    if len(mean) == 1:
        return mean[0],var[0]
    else:
        return mean,var
            
class fast_marching_explorer:
    def __init__(self, gridsize, start_node, end_node, X, Y, mean_value=0, obs=[], GP_l=15.0,GP_sv=5.0,GP_sn=0.5,bl_corner=[0,0],*args,**kwargs):
        self.start_node = start_node
        self.end_node = end_node
        
        # create simple GP model
        self.X = X
        self.Y = Y
        self.mean_value = mean_value
        self.GP_model = GPy.models.GPRegression(X,Y-self.mean_value,GPy.kern.RBF(2))
        self.GP_model.kern.lengthscale = GP_l
        self.GP_model.kern.variance = GP_sv
        self.GP_model.Gaussian_noise.variance = GP_sn
        
        # create cost graph from the GP estimate
        self.GP_cost_args = args
        self.GP_cost_kwargs = kwargs        
        self.GP_cost_graph = fm_graphtools.CostmapGridFixedObs(gridsize[0], gridsize[1], obstacles=obs, bl_corner=bl_corner)
        self.cmodel = mat_cost_function_GP(self.GP_cost_graph, 
            cost_fun=self.GP_model.predict, mean_value=self.mean_value, *self.GP_cost_args, **self.GP_cost_kwargs)
        self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        self.GP_cost_graph.var_fun = self.cmodel.calc_var
                
        #Xtemp, Ytemp = np.meshgrid(np.arange(self.GP_cost_graph.width), np.arange(self.GP_cost_graph.height))
        #self.Xfull = np.vstack([Xtemp.ravel(), Ytemp.ravel()]).transpose()
        #self.Yfull, self.varYfull = self.GP_model.predict(self.Xfull)
        #self.Yfull += mean_value
        #self.cmodel = mat_cost_function(self.GP_cost_graph)
        #self.cmodel.mat = np.reshape(self.Yfull, (self.GP_cost_graph.height, self.GP_cost_graph.width)).transpose()
        #self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        
        self.fbFM = fast_marcher.FullBiFastMarcher(self.GP_cost_graph)
        self.fbFM.set_start(self.start_node)
        self.fbFM.set_goal(self.end_node)
        
        
    def cost_update(self, cost_update,**kwargs):
        self.fbFM.update(cost_update,**kwargs)
        return self.fbFM.updated_min_path_cost
    
    def cost_update_new(self, cost_update,**kwargs):
        self.fbFM.update_new3(cost_update,**kwargs)
        return self.fbFM.updated_min_path_cost
        
    def add_observation(self, Xnew=None, Ynew=None):
        if Xnew is not None and Ynew is not None:
            self.X = np.append(self.X, Xnew, axis=0)
            self.Y = np.append(self.Y, Ynew, axis=0)
        self.GP_model.set_XY(self.X, self.Y-self.mean_value)
        
        self.cmodel = mat_cost_function_GP(self.GP_cost_graph, 
            cost_fun=self.GP_model.predict, mean_value = self.mean_value, *self.GP_cost_args, **self.GP_cost_kwargs)
        #self.Yfull, self.varYfull = self.GP_model.predict(self.Xfull)
        #self.Yfull += self.mean_value
        #self.cmodel.mat = np.reshape(self.Yfull, (self.GP_cost_graph.height, self.GP_cost_graph.width)).transpose()
        self.GP_cost_graph.cost_fun = self.cmodel.calc_cost
        self.GP_cost_graph.var_fun = self.cmodel.calc_var

        self.GP_cost_graph.clear_delta_costs()
        
        self.fbFM.set_graph(self.GP_cost_graph)
        self.fbFM.set_start(self.start_node)
        self.fbFM.set_goal(self.end_node)
        self.search()

    def set_plots(self, imf, ax):
        self.fbFM.set_plots(imf, ax)
        
    def set_plot_costs(self, startcost, delta_cost):
        self.fbFM.set_plot_costs(startcost, delta_cost)
        
    def find_corridor(self):
        self.fbFM.find_corridor()
        
    def search(self):
        # Initial search
        self.fbFM.search()
        self.fbFM.pull_path()
        
    def set_sampler(self, sampler=None):
        self.non_obstacles = []
        for x in range(self.GP_cost_graph.width):
            for y in range(self.GP_cost_graph.height):
                if (x,y) not in self.GP_cost_graph.obstacles:
                    self.non_obstacles.append((x,y))
        self.reset_sampler()
        if sampler == 'random':
            self.sampler = self.random_sampler
        elif sampler == 'fmex':
            self.sampler = self.fmex_sampler
        elif sampler == 'maxvar':
            self.sampler = self.maxvar_sampler
        elif sampler == 'lcb':
            self.sampler = self.lcb_sampler
        else:
            raise ValueError('Unknown sampler specified: {0}'.format(sampler))
        
    def reset_sampler(self):
        self.sampler_u_min = None
        self.best_sample = []
    
    def run_counted_sampler(self, n_samples=None, *args, **kwargs):
        self.reset_sampler()
        if n_samples is None or n_samples > len(self.non_obstacles):
            sample_points = self.non_obstacles
        else:
            sample_points = random.sample(self.non_obstacles, n_samples)
        self.sampler(sample_points, *args, **kwargs)
        return self.best_sample
    
    def random_sampler(self, points):
        self.best_sample = random.sample(points, 1)[0]
        
    def maxvar_sampler(self, points):
        for x,y in points:
            u_sample = self.GP_cost_graph.var_fun(x,y)
            # NOTE THE MAX IS THE GOAL IN THIS CASE!!
            if u_sample > self.sampler_u_min or self.sampler_u_min is None:
                self.sampler_u_min = u_sample
                self.best_sample = [x,y]
    
    def lcb_sampler(self, points, cweight=1.0):
        for x,y in points:
            u_sample = self.GP_cost_graph.cost_fun(x,y)-cweight*math.sqrt(self.GP_cost_graph.var_fun(x,y))
            if u_sample < self.sampler_u_min or self.sampler_u_min is None:
                self.sampler_u_min = u_sample
                self.best_sample = [x,y]        
        
    def fmex_sampler(self, points, cost_obj, delta_costs, weights=None):
        if weights == None:
            weights = [normpdf(dc, 0.0, 1.0) for dc in delta_costs]
            
        for x,y in points:
            try:
                std = math.sqrt(self.GP_cost_graph.var_fun(x,y))
                path_cost,sum_weight = 0.0,0.0
                for dc,weight in zip(delta_costs,weights):
                    path_cost += weight*self.cost_update_new(cost_obj.set_update(x, y, dc*std))
                    sum_weight += weight
                u_sample = path_cost/sum_weight
                if u_sample < self.sampler_u_min or self.sampler_u_min is None:
                    self.sampler_u_min = u_sample
                    self.best_sample = [x,y]
            except TypeError:
                "Caught a TypeError at location ({0},{1})".format(x,y)
                