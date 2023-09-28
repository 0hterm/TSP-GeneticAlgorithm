import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt

class City:
    def __init__(self,x,y):
        self.x = float(x)
        self.y = float(y)

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'
    
class Fitness:
    def __init__(self,route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i+1 < len(self.route):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness
    
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns = ['Index', 'Fitness'])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum/df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []

    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])

    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)

    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1

    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)

    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations, stagnation=False, stagnationThresh = 0):
    pop = initialPopulation(popSize, population)
    print('Initial distance: ' + str(1 / rankRoutes(pop)[0][1]))
    stagnationCounter = 0
    bestRoute = float('inf')
    genCounter = 1
    progress = []

    if stagnation == True:
        while stagnationCounter < stagnationThresh:

            if genCounter > generations:
                print('\nBest distance: {}'.format(bestRoute))
                print('Generation with best distance: {}'.format(bestGen))
                bestRouteIndex = rankRoutes(pop)[0][0]
                bestRoute = pop[bestRouteIndex]
                return bestRoute
            
            print('Generation {}| Minimum Distance: {}'.format(genCounter, 1 / rankRoutes(pop)[0][1]))
            currentBest = 1 / rankRoutes(pop)[0][1]
            progress.append(currentBest)
            
            if currentBest < bestRoute:
                bestRoute = currentBest
                stagnationCounter = 0
                bestGen = genCounter
            else:
                stagnationCounter += 1

            genCounter += 1
            pop = nextGeneration(pop, eliteSize, mutationRate)

        

        print('\nBest distance: {}'.format(bestRoute))
        print('Generation with best distance: {}'.format(bestGen))
        bestRouteIndex = rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        plt.plot(progress)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()
        return bestRoute
    
    else:
        for i in range(0, generations):
            currentBest = 1 / rankRoutes(pop)[0][1]
            progress.append(currentBest)
            print('Generation {}| Minimum Distance: {}'.format(i+1, currentBest))
            
            if currentBest <= bestRoute:
                bestRoute = currentBest
                bestGen = i+1
            pop = nextGeneration(pop, eliteSize, mutationRate)

        print('\nBest distance: ' + str(bestRoute))
        print('Generation with best distance: {}'.format(bestGen))
        bestRouteIndex = rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        plt.plot(progress)
        plt.ylabel('Distance')
        plt.xlabel('Generation')
        plt.show()
        return bestRoute

cityList = []
cityIndices = []

numCities = int(input('Enter the integer number of cities you would like your salesman to travel to: '))
populationSize = int(input('Enter the integer number of the population size (higher population means longer runtime): '))
mutationRate = float(input('Enter float value mutation rate (0.01 will give best results): '))
elitePop = int(input('Enter percent of new children in each generation (enter as a positive integer): '))
genSize = int(input('Enter the numbers of generations you would like to run: '))
stagnation = str(input('Would you like to use stagnation as a stopping condition? (Y/N): '))

while stagnation != 'Y' and stagnation != 'N':
    stagnation = str(input('You must enter \'Y\' or \'N\': '))

if stagnation == 'Y':
    stagnation = True
    stagnationThresh = int(input('Enter number of consecutive generations without improvement (used as stopping condition): '))
else:
    stagnation = False
    stagnationThresh = 0

for i in range(0, numCities):
    cityPoint = City(x=int(random.random() * 200), y=int(random.random() * 200))
    cityList.append(cityPoint)
    cityIndices.append(i)


bestRouteIndices = geneticAlgorithm(population=cityList, popSize=populationSize, eliteSize=elitePop, mutationRate=mutationRate, generations = genSize, stagnation = stagnation, stagnationThresh = stagnationThresh)
finalRouteIndices = [cityList.index(city)for city in bestRouteIndices]
finalRouteCityNumbers = [i + 1 for i in finalRouteIndices]
print('Best Sequence of cities:')
for i, city_number in enumerate(finalRouteCityNumbers, start = 1):
    print("{}. {}".format(i, city_number))
    

