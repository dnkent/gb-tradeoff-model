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
    def __init__(self, pos, model, econ_start, econ_growth, domestic_need, arms, num_adversaries, expenditures):
        '''
        Create a new state.

        Args:
            x, y: Initial location.
            arms: level of military capacity
            econ: economy size
            growth: economic growth
            domestic: baseline needs
            arms: starting arms
            mil_burden: percent of economy spent on military
            adversaries: how many neighbors to consider when balancing
            expenditures: percent of rents to go to programs
        '''
        ## What are the available parameters of interest for each agent?
        ## Need to be specified at the start
        super().__init__(pos, model)
        self.pos = pos
        self.econ = econ_start
        self.growth = econ_growth
        self.domestic = domestic_need
        self.expenditures = expenditures
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

        # Which neighbor is most threatening?
        neighbor_threat = []

        # Survey each neighbor's spending
        for neighbor in self.model.grid.neighbor_iter(self.pos):
            neighbor_threat.append(neighbor.arms)
        
        neighbor_threat = np.array(neighbor_threat)
        # How many states to consider as a threat?
        threat = neighbor_threat.argsort()[-adversaries:]

        # How much can the agent spend on its defense?
        #necessary_costs = (self.econ * self.domestic) + self.arms
        #available = self.econ - necessary_costs
        domestic_costs = self.econ * self.domestic
        available = domestic_costs * (1 - self.expenditures) - self.arms
        #available = self.econ * self.domestic

        # Difference between neighbor arms and agent's current arms?
        threat_diff = np.array(threat) - available

        # Now balance against the biggest state which can be feasibly balanced against -- the smallest negative number
        pot_bal = threat_diff[np.where(threat_diff < 0)]

        # if nobody balance against, then need to denote
        if len(pot_bal > 0):
            balance_cost = np.absolute(np.max(pot_bal))
        else:
            balance_cost = 0
        
        # add to arms
        self.arms = self.arms + balance_cost
        # remove from arms
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
    domestic_min, domestic_max, 
    num_adversaries, expenditures):
        '''
        '''

        self.height = height
        self.width = width
        self.density = density
        self.domestic_min = domestic_min
        self.domestic_max = domestic_max
        self.num_adversaries = num_adversaries
        self.expenditures = expenditures

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
                ## Grow at 3%
                econ_growth = 0.03
                # domestic need -- determines variation in 
                # agent economy
                domestic_need = np.random.uniform(
                    self.domestic_min,
                    self.domestic_max
                    )
                expenditures = self.expenditures
                # starting percent of wealth spent on weapons
                arms_start_perc = np.random.uniform(0, 0.06) 
                arms = arms_start_perc * econ_start

                # create agent
                agent = state((x, y), self, econ_start = econ_start,
                            econ_growth = econ_growth, arms = arms,
                            domestic_need = domestic_need,
                            num_adversaries = num_adversaries,
                            expenditures = expenditures)

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
