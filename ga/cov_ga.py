import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# GA Parameters
POP_SIZE = 50
NGEN = 20
TOURN_SIZE = 3
CXPB = 0.7
MUTPB = 0.2
MUT_STEP = 0.05
RISK_FREE_RATE = 0.0


def random_portfolio(num_stocks):
    w = [random.random() for _ in range(num_stocks)]
    total = sum(w)
    return [x/total for x in w]

def normalize(ind):
    s = sum(ind)
    if s > 0:
        for i in range(len(ind)):
            ind[i] /= s
    else:
        # If sum is zero (extreme case), evenly distribute weights
        for i in range(len(ind)):
            ind[i] = 1.0/len(ind)

def evaluate(individual, expected_returns, cov_matrix):
    # Portfolio return
    w = np.array(individual)
    port_return = w.dot(expected_returns)
    
    # Portfolio variance and std dev using covariance matrix: w^T * Cov * w
    port_variance = w.dot(cov_matrix).dot(w)
    port_vol = np.sqrt(port_variance) if port_variance > 0 else 0.000001

    sharpe = (port_return - RISK_FREE_RATE) / port_vol
    return sharpe

def tournament_selection(pop, fits, tourn_size):
    chosen = random.sample(range(len(pop)), tourn_size)
    best = chosen[0]
    for c in chosen[1:]:
        if fits[c] > fits[best]:
            best = c
    return pop[best][:]

def crossover(ind1, ind2):
    size = len(ind1)
    for i in range(size):
        alpha = random.random()
        w1, w2 = ind1[i], ind2[i]
        new_w1 = alpha * w1 + (1 - alpha) * w2
        new_w2 = alpha * w2 + (1 - alpha) * w1
        ind1[i], ind2[i] = new_w1, new_w2
    normalize(ind1)
    normalize(ind2)

def mutate(ind):
    for i in range(len(ind)):
        if random.random() < MUTPB:
            ind[i] += random.uniform(-MUT_STEP, MUT_STEP)
    # Remove negatives and normalize
    ind[:] = [max(0, w) for w in ind]
    normalize(ind)

def run_ga(expected_returns, cov_matrix, symbols=None):

    if not symbols:
        symbols = cov_matrix.columns
        
    num_stocks = len(expected_returns)
    population = [random_portfolio(num_stocks) for _ in range(POP_SIZE)]
    
    best_individual = None
    best_fitness = float('-inf')
    best_fits_over_time = []
    
    for gen in range(NGEN):
        # Evaluate population
        fitnesses = [evaluate(ind, expected_returns, cov_matrix) for ind in population]

        # Track best of this generation
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx][:]

        best_fits_over_time.append(best_fitness)
        
        # Selection and reproduction
        new_population = []
        for _ in range(POP_SIZE // 2):
            parent1 = tournament_selection(population, fitnesses, TOURN_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURN_SIZE)
            
            if random.random() < CXPB:
                crossover(parent1, parent2)
            
            mutate(parent1)
            mutate(parent2)
            
            new_population.append(parent1)
            new_population.append(parent2)
        
        population = new_population
    
    # Final check
    fitnesses = [evaluate(ind, expected_returns, cov_matrix) for ind in population]
    final_best_idx = np.argmax(fitnesses)
    if fitnesses[final_best_idx] > best_fitness:
        best_fitness = fitnesses[final_best_idx]
        best_individual = population[final_best_idx][:]

    # Print best portfolio with symbols
    print("Best Fitness (Sharpe):", best_fitness)
    print("Best Portfolio Allocation:")
    for sym, w in zip(symbols, best_individual):
        print(f"{sym}: {w:.4f}")
    
    return best_individual, best_fitness, best_fits_over_time