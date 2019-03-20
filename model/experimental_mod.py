import random
import math
import numpy as np

from scipy.stats import truncnorm

from mesa import Model, Agent
from mesa.time import RandomActivation, SimultaneousActivation
## https://mesa.readthedocs.io/en/master/apis/time.html
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector


class state(Agent):
    def __init__(self, pos, model, econ_start, econ_growth, domestic, arms, num_adversaries, expend):
        '''
        Create a new state.

        Args:
            x, y: Initial location.
            arms: level of military capacity
            growth: economic growth
            domestic: baseline needs
            arms: starting arms
            mil_burden: percent of economy spent on military
            adversaries: how many neighbors to consider when balancing
        '''
        ## What are the available parameters of interest for each agent?
        ## Need to be specified at the start
        super().__init__(pos, model)
        self.pos = pos
        self.econ = econ_start
        self.growth = econ_growth
        self.domestic = domestic
        self.expend = expend
        self.arms = arms
        self.mil_burden = self.arms/self.econ
        self.adversaries = num_adversaries 


    ## Step: What is the decision rule each time an agent is selected?
    def step(self):

        # number of neighbors to consider for balancing?
        adversaries = self.adversaries

        # Economic growth
        econ_gains = self.growth * self.econ
        self.econ = self.econ + econ_gains

        # Percent of gdp available to extract?
        domestic = self.domestic

        # Which neighbor is most threatening?
        neighbor_threat = []

        # Survey each neighbor's spending
        for neighbor in self.model.grid.neighbor_iter(self.pos):
            neighbor_threat.append(neighbor.arms)
        
        # How many states to consider as a threat?
        # Return top n most powerful neighbors' military size
        # return indices for n most powerful neighbors
        neighbor_threat = np.array(neighbor_threat)
        threat_ind = neighbor_threat.argsort()[adversaries:]
        # return the actual amount of power projected by neighbors
        threat = neighbor_threat[threat_ind]

        # Turn into np object
        #neighbor_threat = np.array(neighbor_threat)

        # How much can the agent spend on its defense?
        available_rent =  np.around(
            self.econ * domestic, decimals = 0
            )
        # Amount of available rent spent on domestic expenditures
        dom_exp = available_rent * self.expend
        # What is left for military spending?
        available = available_rent - dom_exp
                
        #necessary_costs = (self.econ * self.domestic) + self.arms
        #necessary_costs = (self.econ * self.domestic)
        
        #available = self.econ - necessary_costs
        #available = self.econ * self.domestic

        ########################
        ## Balancing Decision ##
        ########################

        # State considers all neighbors and balances against the biggest one that it can

        # Difference between neighbor arms and agent's current arms?
        threat_diff = np.array(threat) - available

        # Now balance against the biggest state which can be feasibly balanced against -- the smallest negative number
        pot_bal = threat_diff[np.where(threat_diff < 0)]

        # Is there anybody to balance against?
        if len(pot_bal > 0):
            balance_cost = np.absolute(np.max(pot_bal))
        else:
            balance_cost = 0
        
        # add to arms
        # Balance
        self.arms = self.arms + balance_cost
        # remove from econ
        self.econ = self.econ - balance_cost

        ## Round to hundreth to keep decimals from getting out of control
        self.econ = np.around(self.econ, decimals = 2)
        self.arms = np.around(self.arms, decimals = 2)
        self.mil_burden = self.arms/self.econ

class EconMod(Model):
    '''
    Model class for arming model.
    '''

    def __init__(self, height, width, density, 
    #domestic_min, domestic_max, num_adversaries,
    num_adversaries, pareto_scale, domestic, expend):
        '''
        '''

        self.height = height
        self.width = width
        self.density = density
        #self.domestic_min = domestic_min
        #self.domestic_max = domestic_max
        self.expend = expend
        #self.domestic = domestic ## average domestic lid
        self.domestic = domestic ## upper bound on distribution
        self.num_adversaries = num_adversaries

        self.schedule = RandomActivation(self)  # All agents act at once
        self.grid = SingleGrid(height, width, torus=True)
        self.datacollector = DataCollector(
            # Collect data on each agent's arms levels
            agent_reporters={"Arms": "arms",
                            "Military_Burden": "mil_burden",
                            "Econ": "econ",
                            "Domestic": "domestic",
                            "Expend": "expend"})

        # Set up agents
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            if random.random() < self.density:
                ## Set starting economy 
                econ_start = math.ceil(
                    np.random.pareto(pareto_scale)
                    )  + 10
                ## Grow around 3%
                econ_growth = 0.01 * truncnorm.rvs(1.5, 6, 1)
                lower = 0.04
                upper = self.domestic
                mu = (upper + lower)/2
                sigma = 0.05 ## based on real dist
                domestic_need = truncnorm.rvs(
                    (lower - mu) / sigma, (upper - mu) / sigma, 
                    loc = mu, scale = sigma
                    )
                #domestic_need = self.domestic
                expend = self.expend 
                #domestic_need = np.random.uniform(
                #    self.domestic_min,
                #    self.domestic_max
                #    )
                # starting percent of wealth spent on weapons
                arms_start_perc = 0.05 
                arms = arms_start_perc * econ_start

                # create agent
                agent = state(
                    (x, y), 
                    self, 
                    econ_start = econ_start,
                    econ_growth = econ_growth, 
                    arms = arms,
                    domestic = domestic_need,
                    expend = expend,
                    num_adversaries = num_adversaries
                    )

                # place agent in grid
                self.grid.position_agent(agent, (x, y))
                # add schedule
                self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        '''
        Run one step of the model.
        '''
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)
