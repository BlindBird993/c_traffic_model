import random
from collections import defaultdict
from mesa.time import *

class CustomBaseSheduler(BaseScheduler):
    agents_dict = defaultdict(list)
    def __init__(self,model):
        super().__init__(model)

        self.agents_dict = defaultdict(list)

    def add(self,agent):
        self.agents.append(agent)
        agent_class = type(agent)
        self.agents_dict[agent_class].append(agent)

    def remove(self, agent):

        while agent in self.agents:
            self.agents.remove(agent)

        agent_class = type(agent)
        while agent in self.agents_dict[agent_class]:
            self.agents_dict[agent_class].remove(agent)


    def step(self, by_breed=True):

        if by_breed:
            for agent_class in self.agents_dict:
                self.step_breed(agent_class)
            self.steps += 1
            self.time += 1
        else:
            super().step()

    def step_breed(self, breed):

        agents = self.agents_dict[breed]
        for agent in agents:
            agent.step()
        print("Step done!")

    def get_breed_count(self, breed_class):
        return len(self.agents_dict[breed_class])
