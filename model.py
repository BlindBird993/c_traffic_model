from mesa import Agent, Model
from schedule import *
from agents import *
from mesa.datacollection import DataCollector#Data collector

class ConceptModel(Model):
    """A model with some number of agents."""

    number_of_car_agents = 100
    number_of_bar_agents = 2
    number_of_toll_agents = 1

    verbose = True  # Print-monitoring

    def __init__(self, number_of_car_agents = 100,number_of_bar_agents = 2,number_of_toll_agents = 1):

        #set params
        self.number_of_car_agents = number_of_car_agents
        self.number_of_bar_agents = number_of_bar_agents
        self.number_of_toll_agents = number_of_toll_agents


        self.schedule = CustomBaseSheduler(self)
        self.datacollector = DataCollector(
            {"Car agents": lambda m: m.schedule.get_breed_count(CarAgent)})

        #gate agent
        for i in range(self.number_of_toll_agents):
            gate_agent = GateAgent("Gate "+str(i),self,40)
            self.schedule.add(gate_agent)

        #bar agents
        for i in range(self.number_of_bar_agents):
            bar_agent = BarAgent("Bar "+str(i),self,20)
            self.schedule.add(bar_agent)

        #car agents
        for i in range(self.number_of_car_agents):
            car_agent = CarAgent("Car "+str(i), self)
            self.schedule.add(car_agent)

        #trade
        control_agent = ControlAgent("Control Agent",self)
        self.schedule.add(control_agent)

    def step(self):
        """Advance model by one step"""
        self.schedule.step()
        """Collect data"""
        self.datacollector.collect(self)
        if self.verbose:
            print([self.schedule.time,
                   self.schedule.get_breed_count(CarAgent)])

    def run_model(self,step_count = 7):

        for i in range(step_count):
            print("Step {}".format(i))
            for j in range(24):
                self.step()

