from mesa import Agent, Model
import random
import numpy as np
import collections as col
import operator
import copy

class ControlAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)

        self.hour = 0
        self.day = 0
        self.week = 0

        self.stepCount = 0

        self.historyDemands = []
        self.historyProductions = []

        self.distributedDemands = []
        self.summedDemands = []

        self.buyerPriceList = []

        self.GateDistribution = []
        self.Bar1Distribution = []
        self.Bar2Distribution = []

        self.buyers = []
        self.sellers = []
        self.waitCars = []
        self.emergency = []

        self.waitList = []
        self.waitListEmergency = []

        self.testListOfWaitingCars = []

        self.totalSupply = 0
        self.totalDemand = 0
        self.buyerNumber = 0

        self.demands = []
        self.demandPrice = []
        self.supplyPrice = []
        self.clearPrice = 0

        self.priceLimit  = 0

        self.totalCapacity = 0
        self.listOfConsumers = None
        self.dictOfConsumers = None
        self.dictOfWaiting = None

        self.numberOfBuyers = 0
        self.numberOfEmergency = 0
        self.numberOfSellers = 0
        self.numberOfConsumers = 0
        self.numberOfWaitingCars = 0


    def openBars(self, dencity, max_capacity):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, BarAgent) and agent.readyToSell is False and agent.open is False):
                agent.open = np.random.choice([True, False], p=[dencity / max_capacity, 1 - (dencity / max_capacity)])
                agent.emergencyOpen = agent.open
                agent.readyToSell = agent.open
                print("Agent {} is open: {}".format(agent.unique_id, agent.readyToSell))

    def checkMainRoad(self):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) and agent.readyToSell is True):
                self.openBars(agent.dencity,agent.max)

    def getRoadSituation(self):
        road_list = []
        valueList = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                if agent.open is True and agent.readyToSell is True:
                    road_list.append((agent.unique_id,agent.currentSpeed))
                    valueList.append(agent.currentSpeed)

        max_choice = max(valueList)
        choiceIndex = valueList.index(max_choice)
        for index, elem in enumerate(self.sellers):
            if index == choiceIndex:
                max_choice = elem
        print("Available choices {}".format(road_list))
        print("Suggested choice {}".format(max_choice))
        return max_choice

    def getAvailableCapacity(self):
        self.totalCapacity = 0
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                if agent.open is True and agent.readyToSell is True:
                    self.totalCapacity += agent.maxCapacity
        return self.totalCapacity

    def getConsumerDict(self):
        self.listOfConsumers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent,CarAgent) and agent.readyToBuy is True):
                if agent.type != 'emergency':
                    self.listOfConsumers.append((agent.unique_id,agent.price))

        self.dictOfConsumers = dict(self.listOfConsumers)
        print("Consumers {}".format(self.dictOfConsumers))

    def getSellers(self):
        self.numberOfSellers = 0
        self.sellers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                if agent.open is True and agent.readyToSell is True:
                    self.numberOfSellers += 1
                    self.sellers.append(agent.unique_id)
                    print("Sellers {}".format(agent.unique_id))
        print("List of Sellers {}".format(self.sellers))
        print("Number of sellers {}".format(self.numberOfSellers))

    def getBuyres(self):
        self.numberOfBuyers = 0
        self.buyers = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, CarAgent) and agent.readyToBuy is True):
                if agent.type != 'emergency':
                    self.numberOfBuyers += 1
                    self.buyers.append(agent.unique_id)
                    print("Buyers {}, Type {}".format(agent.unique_id, agent.type))
        print("List of Buyers {}".format(self.buyers))
        print("Number of buyers {}".format(self.numberOfBuyers))
        return self.numberOfBuyers


    def getWaitingCars(self):
        self.numberOfWaitingCars = 0
        self.waitCars = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, CarAgent) and agent.isWaiting):
                print("Agent {} status {}".format(agent.unique_id,agent.isWaiting))
                self.numberOfWaitingCars += 1
                self.waitCars.append(agent.unique_id)

        print("List of Cars Waited {}".format(self.waitCars))
        print("Number of Cars Waited {}".format(self.numberOfWaitingCars))

    def getEmergencyCars(self):
        self.numberOfEmergency = 0
        self.emergency = []
        for agent in self.model.schedule.agents:
            if (isinstance(agent, CarAgent) and agent.readyToBuy is True):
                if agent.type == 'emergency':
                    self.numberOfEmergency += 1
                    self.emergency.append(agent.unique_id)
        print("Number of emergency services {}".format(self.numberOfEmergency))

    def getRoadsEmergency(self,buyer):
        road_choice = self.getRoadSituation()
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent,BarAgent)):
                if agent.open is True and agent.readyToSell is True:
                    if agent.unique_id == road_choice:
                        agent.queue.append(buyer.unique_id)
                        agent.dencity += 1
                        agent.maxCapacity -= 1
                        agent.calculateSpeed()

                        agent.queue.append(buyer.unique_id)

                        if isinstance(agent, GateAgent):
                            self.openBars(agent.dencity, agent.max)

                        if agent.maxCapacity <= 0:
                            agent.readyToSell = False
                            self.numberOfSellers -= 1
                            self.sellers.remove(agent.unique_id)

                        buyer.isWaiting = False
                        buyer.readyToBuy = False
                        buyer.waitingTime = 0
                        self.numberOfEmergency -= 1
                        print("Emergency Buyer {}, Ready to Buy {}".format(buyer.unique_id,buyer.readyToBuy))
                        if len(self.emergency) > 0:
                            self.emergency.remove(buyer.unique_id)
                        print("Emergency list {}".format(self.emergency))

    def distributeEmergency(self):
        print("Emergency {}".format(self.emergency))
        while not (self.numberOfEmergency <= 0 or self.numberOfSellers <= 0):
            for agent in self.model.schedule.agents:
                if (isinstance(agent, CarAgent) and agent.readyToBuy is True and agent.type == 'emergency'):
                    print("Agent {}".format(agent.unique_id))
                    print("Number of sellers {}".format(self.numberOfSellers))
                    self.getAvailableCapacity()
                    if self.totalCapacity > 0:
                        self.getRoadsEmergency(agent)
                    else:
                        break
        self.getAvailableCapacity()

        if self.numberOfEmergency > 0 and self.numberOfSellers == 0:
            print("Not enough place")
            self.waitListEmergency = []
            for agent in self.model.schedule.agents:
                if (isinstance(agent, CarAgent) and agent.readyToBuy is True):
                    if agent.type == 'emergency':
                        agent.isWaiting = True
                        self.waitListEmergency.append(agent.unique_id)
            print("Cars left waiting{}".format(self.waitListEmergency))


    def getRoadWaitingCars(self, buyer):
        road_choice = self.getRoadSituation()
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent, BarAgent)):
                if agent.open is True and agent.readyToSell is True:
                    if agent.unique_id == road_choice:
                        agent.queue.append(buyer.unique_id)
                        agent.dencity += 1
                        agent.maxCapacity -= 1
                        agent.calculateSpeed()

                        agent.queue.append(buyer.unique_id)

                        if agent.maxCapacity <= 0:
                            agent.readyToSell = False
                            self.numberOfSellers -= 1
                            self.sellers.remove(agent.unique_id)

                        if isinstance(agent, GateAgent):
                            self.openBars(agent.dencity, agent.max)

                        buyer.isWaiting = False
                        buyer.readyToBuy = False
                        self.numberOfWaitingCars -= 1
                        if len(self.waitCars) > 0:
                            self.waitCars.remove(buyer.unique_id)
                        print("Waiting car {}".format(buyer.unique_id))
                        print("Available capacity {}".format(self.totalCapacity))


    def distributeWaiting(self):
        print("Waiting cars list {}".format(self.waitCars))
        while not (self.numberOfWaitingCars <= 0 or self.numberOfSellers <=0):
            for agent in self.model.schedule.agents:
                if (isinstance(agent, CarAgent) and agent.isWaiting is True):
                        print("Number of sellers {}".format(self.numberOfSellers))
                        self.getAvailableCapacity()
                        if self.totalCapacity > 0:
                            self.getRoadWaitingCars(agent)
                        else:
                            break
        self.getAvailableCapacity()

        if self.numberOfWaitingCars > 0 and self.numberOfSellers == 0:
            print("Not enough place")
            self.waitCars = []
            for agent in self.model.schedule.agents:
                if (isinstance(agent, CarAgent) and agent.readyToBuy is True):
                    agent.isWaiting = True
                    # agent.waitingTime += 1
                    self.waitCars.append(agent.unique_id)
            print("Cars left waiting{}".format(self.waitCars))

    def calculateFitness(self,test_vector):
        #calculate fitness according to vector of consumers, their demands and available production
        fitness = 0
        summedValue = 0.0
        lst = []
        size_param = 0
        x = sorted(self.dictOfConsumers.items(), key=operator.itemgetter(1))
        x.reverse()
        for index,elem in enumerate(test_vector):
            if elem > 0:
                size_param += 1
                lst.append(list(x[index]))
        for elem in lst:
            summedValue += elem[1]
            summedValue = round(summedValue,3)

        if size_param > self.totalCapacity:
            fitness = 0.0
        else:
            fitness = summedValue
        return fitness

    def generatePopulation(self):
        if self.numberOfBuyers > 1:
            popSize = self.numberOfBuyers
        else:
            print("Not enough buyers")
            popSize = self.numberOfBuyers

        vector_list = []
        n, p = 1, 0.5
        d = col.defaultdict(list)

        for i in range(600):
            pop = np.random.binomial(n, p, popSize)
            pop = list(pop)
            self.calculateFitness(pop)
            d[self.calculateFitness(pop)].append(pop)
            vector_list.append((pop,self.calculateFitness(pop)))
        print("Population {}".format(vector_list))
        ordered_dict = col.OrderedDict(sorted(d.items(), key=lambda t: t[0], reverse=True))
        print("Ordered dict {}".format(ordered_dict))
        sorted_x = sorted(ordered_dict.items(), key=operator.itemgetter(0))
        print("Sorted list {}".format(sorted_x))
        if self.numberOfSellers > 0:
            sorted_x.reverse()
        chosen_elem_list = sorted_x[0]
        print("Best element {}".format(list(chosen_elem_list)[1][0]))
        print("Best fitness {}".format(list(chosen_elem_list)[0]))

        if list(chosen_elem_list)[0] <= 0.0:
            print("Unable to satisfy demand!")
            return 0

        dna = list(chosen_elem_list)[1][0]
        print("DNA {}".format(dna))

        #tournament
        tournament_pool = []
        for i in range(5):
            elem = random.choice(list(d.items()))
            tournament_pool.append(elem)
        print("Tournament pool {}".format(tournament_pool))
        tournament_dict = col.OrderedDict(sorted(tournament_pool, key=lambda t: t[1], reverse=True))
        print(tournament_dict)

        mating_partners = list(tournament_dict.items())

        partners_list = []
        for elem in mating_partners:
            partners_list.append(list(elem)[1][0])
        #get dna for mating
        number_of_partners = 2
        partner = partners_list[0]

        for i in range(600):
            coef = np.random.uniform(0, 1, 1)
            if coef > 0.8:
                dna1 = copy.deepcopy(dna)
                mutated_dna = self.mutate(dna1,self.numberOfBuyers)

                fitness_mutated = self.calculateFitness(mutated_dna)
                fitness_old = self.calculateFitness(dna)
                if fitness_mutated > fitness_old:
                    dna = mutated_dna
            else:
                cross_dna1,cross_dna2 = self.crossover(dna,partner,self.numberOfBuyers)
                fitnes_corss1 = self.calculateFitness(cross_dna1)
                fitnes_corss2 = self.calculateFitness(cross_dna2)
                fitness_old = self.calculateFitness(dna)

                if fitnes_corss1 > fitness_old:
                    dna = cross_dna1

                elif fitnes_corss2 > fitness_old:
                    dna = cross_dna2

        print("Chosen DNA {}".format(dna))
        self.decodeList(dna)

    def crossover(self,dna1, dna2, dna_size):
        pos = int(random.random() * dna_size)
        return (dna1[:pos] + dna2[pos:], dna2[:pos] + dna1[pos:])

    def mutate(self,dna,size):
        mutation_chance = 100 #mutation chance
        for index, elem in enumerate(dna):
            if int(random.random() * mutation_chance) == 1:
                if dna[index] == 1:
                    dna[index] = 0
                else:
                    dna[index] = 1
        return dna

    def decodeList(self,dna_variant):

        self.distributedDemands = []
        self.buyerNumber = 0
        buyers_list = list(self.dictOfConsumers.items())
        print(list(buyers_list))
        for index,elem in enumerate(dna_variant):
            if elem > 0:
                self.buyerNumber += 1
                self.distributeCars(list(buyers_list[index]))

    def getRoadDistribution(self,car_agent,road_choice):
        for agent in self.model.schedule.agents:
            if (isinstance(agent, GateAgent) or isinstance(agent, BarAgent)):
                if agent.open is True and agent.readyToSell is True:
                    if agent.unique_id == road_choice:
                        agent.queue.append(car_agent.unique_id)
                        agent.dencity += 1
                        agent.maxCapacity -= 1
                        agent.calculateSpeed()
                        agent.queue.append(car_agent.unique_id)
                        if isinstance(agent, GateAgent):
                            self.openBars(agent.dencity, agent.max)

                        if agent.maxCapacity <=0:
                            agent.readyToSell = False
                            self.numberOfSellers -= 1
                            self.sellers.remove(agent.unique_id)

                        car_agent.isWaiting = False
                        car_agent.readyToBuy = False
                        self.numberOfBuyers -= 1
                        print("Car {}".format(car_agent.unique_id))
                        self.dictOfConsumers = self.removeItem(self.dictOfConsumers, car_agent.unique_id)

    def distributeCars(self,data_list):
        road_choice = self.getRoadSituation()
        for agent in self.model.schedule.agents:
            if (isinstance(agent, CarAgent)):
                if (agent.unique_id == data_list[0] and agent.readyToBuy is True):
                    print("Agent {}".format(agent.unique_id))
                    self.getAvailableCapacity()
                    self.getRoadDistribution(agent,road_choice)
                    self.getAvailableCapacity()
        print("Consumers dictionary {}".format(self.dictOfConsumers))

    def checkIfConsumersLeft(self):
        if self.dictOfConsumers:
            ordered_dict = col.OrderedDict(sorted(self.dictOfConsumers.items(), key=lambda t: t[1],reverse=True))
            self.getAvailableCapacity()
            print("Consumers left {}".format(ordered_dict.items()))
            for k,v in ordered_dict.items():
                for agent in self.model.schedule.agents:
                    if (isinstance(agent, CarAgent)):
                        if (agent.unique_id == k and agent.readyToBuy is True):
                            print("Agent {}".format(agent.unique_id))

                            if self.totalCapacity > 0:
                                road_choice = self.getRoadSituation()
                                self.getRoadDistribution(agent,road_choice)
                                self.buyerNumber += 1
                                self.getAvailableCapacity()
                            else:
                                agent.isWaiting = True

            print("Customers left {}".format(self.dictOfConsumers))

    # delete elements from set of buyers
    def removeItem(self,d,key):
        del d[key]
        return d

    def test_func(self):
        print("Control Agent {}".format(self.unique_id))

    def getNewCars(self):
        newCars = 0
        for agent in self.model.schedule.agents:
            if ((isinstance(agent, CarAgent) and agent.readyToBuy is True and agent.isWaiting is False)):
                if agent.type != 'emergency':
                    newCars += 1
        return newCars

    def getCarDistribution(self):
        for agent in self.model.schedule.agents:
            if ((isinstance(agent, GateAgent) or isinstance(agent,BarAgent))):
                if agent.unique_id == 'Gate 0':
                    self.GateDistribution.append(len(agent.queue))
                elif agent.unique_id == 'Bar 0':
                    self.Bar1Distribution.append(len(agent.queue))
                elif agent.unique_id == 'Bar 1':
                    self.Bar2Distribution.append(len(agent.queue))

    def step(self):
        print("Trade!\nhour {}\nday {}\nweek {}".format(self.hour, self.day, self.week))
        self.test_func()

        self.getSellers()
        self.getAvailableCapacity()

        self.getWaitingCars()
        self.getBuyres()
        self.getConsumerDict()

        self.getRoadSituation()

        if self.numberOfWaitingCars > 0 and self.totalCapacity > 0:
            self.distributeWaiting()
            self.getAvailableCapacity()

        if self.numberOfEmergency > 0 and self.totalCapacity > 0:
            self.getEmergencyCars()
            self.distributeEmergency()
            self.getAvailableCapacity()

        capacityValue = self.getAvailableCapacity()
        self.historyProductions.append(self.totalCapacity)

        demandValue = self.getNewCars()
        self.historyDemands.append(demandValue)

        if self.totalCapacity > 0:
            self.getBuyres()
            self.getConsumerDict()
            self.generatePopulation()
            self.checkIfConsumersLeft()
        else:
            print("Not enough space!")
        self.summedDemands.append(self.buyerNumber)
        self.stepCount += 1
        self.hour +=1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class CarAgent(Agent):
    def __init__(self,unique_id, model):
        super().__init__(unique_id, model)
        self.hour = 0
        self.day = 0
        self.week = 0
        self.statusPriority = None
        self.price = 0
        self.traided = None

        self.goingToPass = None
        self.choice = None

        self.isWaiting = False
        self.waitingTime = 0

        self.priceHistory = []
        self.priorityHistorySell = []
        self.priorityHistoryBuy = []

        self.type = None #type of the car

        self.readyToBuy = True

    def checkWaitingTime(self):
        if self.waitingTime > 0:
            self.isWaiting = True
            print("Waiting time {}".format(self.waitingTime))
            self.waitingTime -= 1

    def name_func(self):
        print("Agent {}".format(self.unique_id))

    def getType(self):
        if not self.isWaiting:
            self.type = np.random.choice(['car','lorry','emergency'],p=[0.6,0.3,0.1])
        print("Type {}".format(self.type))

    def getPassStatus(self): #get probability based on rush hours
        if self.hour >= 7 and self.hour <= 9:
            self.goingToPass = np.random.choice([True,False],p=[0.9,0.1])
        elif self.hour >= 15 and self.hour <= 17:
            self.goingToPass = np.random.choice([True, False], p=[0.9, 0.1])
        else:
            self.goingToPass = np.random.choice([True, False]) #standart distribution
        print("Status {}".format(self.goingToPass))

    def checkIfWaiting(self):
        if self.isWaiting:
            self.goingToPass = True
        print("Waiting {}".format(self.isWaiting))
        print("Going to Pass {}".format(self.goingToPass))

    def getTradeStatus(self):
        if self.goingToPass:
            self.readyToBuy = True
        else:
            self.readyToBuy = False

    def calculatePrice(self):
        if self.type == 'car':
            self.price = round(random.uniform(46,56),1) #price for getting through, NOK
        elif self.type == 'lorry':
            self.price = round(random.uniform(132, 162), 1)
        else:
            self.price = 0
        print("Price {}".format(self.price))

    def step(self):
        self.name_func()
        self.getType()
        self.getPassStatus()
        self.checkIfWaiting()
        self.getTradeStatus()
        self.calculatePrice()
        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class GateAgent(Agent):
    def __init__(self,unique_id, model,max_capacity=4,road_lenght = 7):
        super().__init__(unique_id, model)
        self.queue = []
        self.open = True
        self.readyToSell = True

        self.initialSpeed = 70
        self.currentSpeed = 0

        self.timeOfTravel = 0

        self.initial_step = True

        self.hour = 0
        self.day = 0
        self.week = 0

        self.price = 0
        self.pricelorry = 0

        self.roadLength = road_lenght
        self.speedLimit = 0

        self.initialPropensityValuesCar = []
        self.initialPropensityValueslorry = []

        self.rushHour = False
        self.number_of_cars = 0
        self.dencity = 0
        self.max = max_capacity
        self.maxCapacity = max_capacity

        self.choicelorry = 0
        self.stateChoicelorry = None

        self.currentState = None

    def calculateSpeed(self):
        n = 2
        m = 2
        speed_val = self.initialSpeed*((1-self.dencity/self.max)**m)**n
        self.currentSpeed = round(speed_val,3)
        if self.currentSpeed <= 0:
            self.timeOfTravel = self.roadLength*10
        else:
            self.timeOfTravel = self.roadLength/self.currentSpeed
            self.timeOfTravel = round(self.timeOfTravel*10,3)
        print("Speed value {}".format(self.currentSpeed))
        print("Time value {}".format(self.timeOfTravel))

    def checkIfReadyToSell(self):
        if self.open:
            self.readyToSell = True
        print("Ready to Sell {}".format(self.readyToSell))
        print("Capacity {}".format(self.maxCapacity))

    def updateQueue(self):
        self.queue = []
        self.dencity = 0
        self.maxCapacity = self.max

    def checkIfRushHour(self):
        if self.hour >= 7 and self.hour <= 9:
            self.rushHour = True
        elif self.hour >= 15 and self.hour <= 17:
            self.rushHour = True
        else:
            self.rushHour = False
        print("Rush Hour {}".format(self.rushHour))

    def name_func(self):
        print("Agent {}, length {}".format(self.unique_id,self.roadLength))

    def step(self):
        self.name_func()
        self.updateQueue()
        self.checkIfReadyToSell()
        self.checkIfRushHour()
        self.calculateSpeed()

        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0

class BarAgent(Agent):
    def __init__(self,unique_id, model, max_capacity = None,road_length = 10):
        super().__init__(unique_id, model)
        self.open = True

        self.initialSpeed = 60
        self.currentSpeed = 0
        self.timeOfTravel = 0

        self.queue = []
        self.readyToSell = True

        self.initial_step = True

        self.emergencyOpen = False

        self.hour = 0
        self.day = 0
        self.week = 0

        self.price = 0
        self.pricelorry = 0

        self.roadLength = road_length
        self.speedLimit = 0

        self.number_of_cars = 0
        self.dencity = 0
        self.max = max_capacity
        self.maxCapacity = max_capacity

        self.currentState = None

    def calculateSpeed(self):
        n = 2
        m = 2
        speed_val = self.initialSpeed*((1-self.dencity/self.max)**m)**n
        self.currentSpeed = round(speed_val,3)
        if self.currentSpeed <= 0:
            self.timeOfTravel = self.roadLength*10
        else:
            self.timeOfTravel = self.roadLength/self.currentSpeed
            self.timeOfTravel = round(self.timeOfTravel*10,3)
        print("Speed value {}".format(self.currentSpeed))
        print("Time value {}".format(self.timeOfTravel))

    def checkIfReadyToSell(self):
        if self.open:
            self.readyToSell = True
        print("Ready to Sell {}".format(self.readyToSell))
        print("Capacity {}".format(self.maxCapacity))

    def updateQueue(self):
        self.queue = []
        self.dencity = 0
        self.maxCapacity = self.max

    def setStatus(self):
        if self.hour >= 7 and self.hour <= 9 and self.emergencyOpen is False:
            self.open = False
            self.readyToSell = False
        else:
            self.open = True
            self.readyToSell = True
            self.emergencyOpen = False
        print("Status {}".format(self.open))

    def checkIfRushHour(self):
        if self.hour >= 7 and self.hour <= 9:
            self.rushHour = True
        elif self.hour >= 15 and self.hour <= 17:
            self.rushHour = True
        else:
            self.rushHour = False
        print("Rush Hour {}".format(self.rushHour))


    def name_func(self):
        print("Agent {}, length {}".format(self.unique_id,self.roadLength))

    def step(self):
        self.name_func()
        self.updateQueue()
        self.setStatus()
        self.checkIfReadyToSell()
        self.checkIfRushHour()
        self.calculateSpeed()

        self.hour += 1

        if self.hour > 23:
            self.day += 1
            self.hour = 0

        if self.day > 7:
            self.week += 1
            self.day = 0