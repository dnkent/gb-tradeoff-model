import random
import math
import numpy as np

from scipy.stats import truncnorm

from mesa import Model, Agent
from mesa.time import RandomActivation, SimultaneousActivation
## https://mesa.readthedocs.io/en/master/apis/time.html
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

## This is the same model as usual, but I am starting all agents as identical economically. So same units economically, but just vary domestic needs.
## Will vary starting arms level

class state(Agent):
    def __init__(self, pos, model, econ_start, econ_growth, domestic_need,
                arms):
        '''
        Create a new state.

        Args:
            x, y: Initial location.
            arms: level of military capacity
            growth: economic growth
            domestic: baseline needs
            arms: starting arms
        '''
        ## What are the available parameters of interest for each agent?
        ## Need to be specified at the start
        super().__init__(pos, model)
        self.pos = pos
        self.econ = econ_start
        self.growth = econ_growth
        self.domestic = domestic_need
        self.arms = arms
        self.mil_burden = self.arms/self.econ ## costs of anarchy
        #self.available = None
        #self.necessary_costs = None
        #self.neighbor = None

    ## Step: What is the decision rule each time an agent is selected?
    def step(self):
        # Economic growth
        ## Numpy has trouble with too many decimals, so let's round up
        ## to the nearest thousandth
        econ_last = self.econ
        econ_gains = self.growth * econ_last
        self.econ = econ_last + econ_gains

        # Sum all arms in state's vicinity
        threat = 0
        ## can vary the radius of neighbors
        ## https://mesa.readthedocs.io/en/master/apis/space.html?highlight=neighbor_iter
        for neighbor in self.model.grid.neighbor_iter(self.pos):
            threat += neighbor.arms

        # Is there enough available $$ after domestic needs are met?
        # If yes, meet threat. If no, just invest available on arms.
        necessary_costs = (self.econ * self.domestic) + self.arms
        available = self.econ - necessary_costs

        if available >= threat:
            self.arms = self.arms + threat
        else:
            self.arms = self.arms + available

        # Don't subtract from economy -- that is how much
        # the country produces in a year which can then be
        # extracted for mil rent
        #if (self.econ - threat) >= available:
        #    self.arms = self.arms + threat
        #    self.econ = self.econ - threat
        #else:
        #    self.arms = self.arms + available
        #    self.econ = self.econ - available

        ## Round to hundreth to keep decimals from getting out of control
        self.econ = np.around(self.econ, decimals = 2)
        self.arms = np.around(self.arms, decimals = 2)
        self.mil_burden = self.arms/self.econ

class EconMod(Model):
    '''
    Model class for arming model.
    '''

    def __init__(self, height, width, density, domestic_min, domestic_max):
        '''
        '''

        self.height = height
        self.width = width
        self.density = density
        self.domestic_min = domestic_min
        self.domestic_max = domestic_max

        self.schedule = RandomActivation(self)  # All agents act at once
        self.grid = SingleGrid(height, width, torus=True)
        self.datacollector = DataCollector(
            # Collect data on each agent's arms levels
            agent_reporters={"Arms": "arms",
                            "Military_Burden": "mil_burden",
                            "Econ": "econ",
                            "Domestic": "domestic"})

        # Set up agents
        for cell in self.grid.coord_iter():
            x = cell[1]
            y = cell[2]
            if random.random() < self.density:
                ## Set starting economy to 10 constant for all
                econ_start = 10
                ## Growth starts at 0.02
                econ_growth = 0.02
                # domestic need: certain percent of wealth -- between 50 and 100%
                domestic_need = np.random.uniform(self.domestic_min,
                                                self.domestic_max)
                # starting percent of wealth spent on weapons
                # arms_start_perc = math.ceil(np.random.uniform(1, 10))
                arms_start_perc = np.random.uniform(0, 0.06) ## start with 5% at most
                #arms = math.ceil(arms_start_perc * econ_start)
                arms = arms_start_perc * econ_start
                # create agent
                agent = state((x, y), self, econ_start=econ_start,
                            econ_growth=econ_growth, arms=arms,
                            domestic_need=domestic_need)
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
