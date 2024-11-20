import random
from venv import create

import numpy as np
from deap import base, creator, tools
A, B = -10, 10
CHROM_LEN = 3
ETA = 20
POPULATION_SIZE = 180
CROSSOVER = 0.85
MUTATION = 0.15
GENERATIONS = 40

def eval_func(ind):
    x, y, z = ind
    return 1/(1 + (x - 2) ** 2 +(y + 1) ** 2 + (z - 1) ** 2),

def random_point(a, b):
    return [random.uniform(a,b), random.uniform(a,b), random.uniform(a,b)]

def toolbox_create():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("randomPoint", random_point, A, B)
    toolbox.register("indCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
    toolbox.register("popCreator", tools.initRepeat, list, toolbox.indCreator)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low = A, up = B, eta = ETA)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mutate", tools.mutPolynomialBounded, low = A, up = B, eta = ETA, indpb = 1.0 / CHROM_LEN)
    stats = tools.Statistics(lambda ind_: ind_.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    return toolbox

toolbox_ = toolbox_create()
random.seed(7)
population = toolbox_.popCreator(n = POPULATION_SIZE)
fitnesses = list(map(toolbox_.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit
print("\nEvaluated", len(population), "individuals")
for gen in range(GENERATIONS):
    print("Generation: ", gen)
    offspring = toolbox_.select(population, len(population))
    offspring = list(map(toolbox_.clone, offspring))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER:
            toolbox_.mate(child1, child2)
            del child1.fitness.values, child2.fitness.values
    for mut in offspring:
        if random.random() < MUTATION:
            toolbox_.mutate(mut)
            del mut.fitness.values
    inv_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox_.evaluate, inv_ind))
    for ind, fit in zip(inv_ind, fitnesses):
        ind.fitness.values = fit
    print("\nEvaluated", len(inv_ind), "individuals")
    population[:] = offspring
    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    print('Min = ', min(fits), ', Max = ', max(fits))
    print('AVG = ', round(mean, 2), ', STD = ', round(std, 2))

    best = tools.selBest(population, 1)[0]
    print('Best = ', best)
    print('Num of ones: ', sum(best))