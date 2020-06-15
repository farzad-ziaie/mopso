# written by: Farzad ziaie nezhad
# farzadziaien@gmail.com
# +989174509365

# refrences:
# https://faradars.org/courses/mvrmo9012-multiobjective-optimization-video-tutorials-pack
# mopso: Coello Coello
# https://ieeexplore.ieee.org/document/1004388




import numpy as np
from tqdm import tqdm, tnrange



import numpy as np

def random_generator(h, w):
    random_positions = []
    for _ in range(h):
        random_positions.append(2*(np.random.rand(w)-.5))
    arr = np.arange(h)    
    np.random.shuffle(arr)    
    indx = arr.astype(np.uint32).tolist()
    mylist = [random_positions[i] for i in indx]
    return mylist
    
class mopso:
    def __init__(self, cost__=None, innerset=2, outerset=np.random.randn(4), varmin=-4, 
                 varmax=4, varsize=3, nvar=3, wdamp=0.99, mu=0.1, npop=150, nrepo=50, 
                 w=0.2, c1=0.1, c2=0.1, beta=.5, gamma=.5, nGrid=10, alpha=.1, debug=False):
        """
        :param cost: Cost Function
        :param innerset:
        :param outerset:
        :param varmin: Lower Bound of Variables
        :param nvar: Number of Decision Variables
        :param vaemax: Upper Bound of Variables
        :param varsize: Size of Decision Variables Matrix
        :param maxiter: Maximum Number of Iterations
        :param npop: Population Size
        :param nself.repo: self.repository Size
        :param w: Inertia Weight
        :param wdamp: Intertia Weight Damping Rate
        :param c1: Personal Learning Coefficient
        :param c2: Global Learning Coefficient
        :param beta: Leader Selection Pressure
        :param gamma: Deletion Selection Pressure
        :param alpha: Inflation Rate
        :param nGrid: Number of Grids per Dimension
        :param mu: Mutation Rate
        :param debug: only for cost function diagnosis
        """
        self.varmin = varmin
        self.w = w
        self.cost_fun = cost__
        self.wdamp = wdamp
        self.c1 = c1
        self.c2 = c2
        self.varmax = varmax
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ngrid = nGrid
        self.nrepo = nrepo
        self.weight_size = len(outerset)
        self.weights = outerset
        self.pop = []
        self.npop = npop
        random_pos = random_generator(npop, self.weight_size)
#         number of objectives, or how many output targets we have
        self.n_obj = 2
        #         check cost function for the number of output dimentions: be sure it's a np.array -> np.array([1,2,3,...])
        if debug == False:
            try:
                OutputCost = self.cost_fun.evaluate(random_pos[0])
                if type(OutputCost) != type(np.array([0])):
                    raise ValueError('the output of the cost function is not correct')
                self.n_obj = OutputCost.shape[0]
            except:
                raise ValueError("there is something wrong within your cost function implementation, for diagnosis you can run in 'debug' mode")
        else:
            self.n_obj  = self.cost_fun.evaluate(random_pos[0]).shape[0]            
            
#         initializations
    # determining a population of particles 
        for i in range(self.npop):
            self.pop.append({
                'position': random_pos[i],
                'velocity': np.zeros(self.weight_size),
                'cost': None,
                'best_position': None,
                'best_cost': None,
                'isdominated': False,
                'gridindex': [],
                'gridsubindex': []
            })
        for i in range(self.npop):
            self.pop[i]['cost'] = self.cost(self.pop[i]['position'])
            self.pop[i]['best_position'] = self.pop[i]['position']
            self.pop[i]['best_cost'] = self.pop[i]['cost']
        
        self.pop = self._pop_determine_domination(pops=self.pop)
        buf_domination = [not self.pop[i]['isdominated'] for i in range(npop)]
        
        self.repo = []
        for i in range(len(buf_domination)):
            if buf_domination[i]:
                self.repo.append(self.pop[i])        
        self.grid = self._CreateGrid(self.repo)
        for i in range(len(self.repo)):
            self.repo[i] = self._find_grid_index(self.repo[i], self.grid)                
        
#         self.fit()

#     main loop
    def fit(self, maxiter=50):        
        for it in tnrange(maxiter, desc = 'progress'):
            if it == 0:
                print('iter', '      costs .....................')
            for i in range(self.npop):
                leader = self._leader_selection(self.repo)
                self._update_velocity_pos(leader, i)              
                
                self.pop = self._pop_determine_domination(pops=self.pop) 
                
                if self._dominates(self.pop[i]['position'], self.pop[i]['best_position']):
                    self.pop[i]['best_position'] = self.pop[i]['position']
                    self.pop[i]['best_cost'] = self.pop[i]['cost']
                elif self._dominates(self.pop[i]['best_position'], self.pop[i]['position']):
                    pass
                else:
                    if np.random.random(1) < 0.5:
                        self.pop[i]['best_position'] = self.pop[i]['position']
                        self.pop[i]['best_cost'] = self.pop[i]['cost']

            # check for the populations' new domination
            buf_domination = [not self.pop[i]['isdominated'] for i in range(self.npop)]
            for i in range(len(buf_domination)):
                if buf_domination[i]:
                    self.repo.append(self.pop[i])        
            self.grid = self._CreateGrid(self.repo)
            for i in range(len(self.repo)):
                self.repo[i] = self._find_grid_index(self.repo[i], self.grid) 

            # check for the repository new domination                    
            rep = self._pop_determine_domination(pops = self.repo)
            buf_domination = [not rep[i]['isdominated'] for i in range(len(rep))]
            self.repo = []
            for i in range(len(buf_domination)):
                if buf_domination[i]:
                    self.repo.append(rep[i])        
            self.grid = self._CreateGrid(self.repo)
            for i in range(len(self.repo)):
                self.repo[i] = self._find_grid_index(self.repo[i], self.grid)                         
            
            # check for the size of the repository and pruning 
            extera = len(self.repo) - self.nrepo
            if extera > 0 :                
                for e in range(extera-1):
                    self.repo = self._detelte_rep_mem(self.repo)
            
            self.w = self.w * self.wdamp
            print(it,'------>', self.repo[0]['cost'])
                    
        
    def _update_velocity_pos(self, leader, i):
        buf1 = self.w * self.pop[i]['velocity'] 
        buf2 = self.c1 * np.random.random(self.weight_size) * (self.pop[i]['best_position'] - self.pop[i]['position'])
        buf3 = self.c2 * np.random.random(self.weight_size) * (leader['best_position'] - self.pop[i]['position'])    
        self.pop[i]['velocity'] = buf1 + buf2 + buf3
        self.pop[i]['position'] = self.pop[i]['position'] + self.pop[i]['velocity']
        self.pop[i]['cost'] = self.cost(self.pop[i]['position'])
        
    def _leader_selection(self, repo):
        grid_index = [rep['gridindex'] for rep in repo]
        occupied_cells = set(grid_index)
        number_of_duplicates = {}
        for num in occupied_cells:
            number_of_duplicates[num] = grid_index.count(num)
        p = {}
        for key, val in number_of_duplicates.items():
            p[key] = np.exp(-self.beta*val)   
#         finding a cell index
        sci = self._roulette_wheel_selection(p)        
#         select cell members
        n = 0
        for gi in grid_index:
            if gi == sci:
                n += 1                
        return repo[np.random.randint(n, size=1)[0]]        

    def _detelte_rep_mem(self, repo):
        grid_index = [rep['gridindex'] for rep in repo]
        occupied_cells = set(grid_index)
        number_of_duplicates = {}
        for num in occupied_cells:
            number_of_duplicates[num] = grid_index.count(num)
        p = {}
        for key, val in number_of_duplicates.items():
            p[key] = np.exp(-self.gamma*val)   
#         finding a cell index
        sci = self._roulette_wheel_selection(p)        
#         select cell members
        n = 0
        for gi in grid_index:
            if gi == sci:
                n += 1   
#         print(len(repo), n, np.random.randint(n, size=1)[0])
        repo.pop(np.random.randint(n, size=1)[0])
        return repo
                  
    def _roulette_wheel_selection(self, p):
        r = np.random.random(1)
        keys = list(p.keys())
        vals = list(p.values())
        vals = vals/np.sum(vals)
        ci = np.cumsum(vals)
        for i in range(ci.shape[0]):
            if r <= ci[i]:                
                return keys[i]
        return keys[0]
            
    def cost(self, x):
        
        if self.cost_fun == None:
#             n = self.weight_size;
    #         z1 = 1-np.exp(-np.sum((x-1/np.sqrt(n))**2));
    #         z2 = 1-np.exp(-np.sum((x+1/np.sqrt(n))**2));
            z1 = np.sum(x ** 2)
            z2 = np.sum(x ** 4)
            z3 = np.sum(x ** 6)
            return np.array([z1, z2, z3])
        else:
            return self.cost_fun.evaluate(x)

    def _pop_determine_domination(self, pops):        
        npop = len(pops)
        for i in range(npop):
            pops[i]['isdominated'] = False

        for i in range(npop-1):
            for j in range(i + 1, npop):                    
                if self._dominates(pops[i]['cost'], pops[j]['cost']):
                    pops[j]['isdominated'] = True
                if self._dominates(pops[j]['cost'], pops[i]['cost']):
                    pops[i]['isdominated'] = True
        return pops
    
    def _dominates(self, x, y):
        x_ind = []
        y_ind = []        
        for ij in range(x.shape[0]):
            x_ind.append(x[ij] <= y[ij])
            y_ind.append(x[ij] < y[ij])        
        return all(x_ind) and any(y_ind)
    
    def _CreateGrid(self, repo):
        """
        creating a sub space in the objective space 
        in order to break it down into pieces for more focused search
        
        self.alpha: inflation rate
        self.grid: number of subsets for cost function output space for each object 
        """
        costs = []
        for pop in repo:
            costs.append(pop['cost'])
        costs = np.array(costs)
        min_ = np.min(costs, axis=0)
        max_ = np.max(costs, axis=0)
        dc = max_ - min_
        max_ = max_ + self.alpha*dc
        min_ = min_ - self.alpha*dc        
            #expanding the grid to inf. 
        Grid = []
        for j in range(self.n_obj):
            cut_points = np.linspace(min_[j] ,max_[j] , self.ngrid+1)
            Grid.append({
                'lower_bands': np.array([-np.inf, *cut_points.tolist()]) ,
                'upper_bound': np.array([*cut_points.tolist(), np.inf])
            })
        return Grid 
    
    def _find_grid_index(self, particle, grid):
        
#         calculating grid subindex
        particle['gridsubindex'] = np.zeros(self.n_obj)
        for j in range(self.n_obj):
#             print(particle['cost'][j], grid[j]['upper_bound'])
            particle['gridsubindex'][j] = Ccompare(particle['cost'][j], grid[j]['upper_bound'])
#             claculating grid index -> a simple mapping 
        particle['gridindex'] = particle['gridsubindex'][0]
        for j in range(1, self.n_obj):
            particle['gridindex'] = particle['gridindex'] - 1
            particle['gridindex'] = particle['gridindex'] * (self.ngrid+2)
            particle['gridindex'] = (particle['gridindex'] + particle['gridsubindex'][j]).astype(np.uint64)
        return particle
            
            
# ---------------------------------------------------------------------------------------------------------- utils            
# utils
def Ccompare(a, b):
#     b: longer vector
#     a: shorter vector -> important one -> the one you want to base your comparison on it       
    for j in range(b.shape[0]):
        if a < b[j]:
            return j
            