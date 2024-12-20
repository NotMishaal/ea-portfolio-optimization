import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticOptimizer:
    def __init__(self, pop_size=50, ngen=20, tourn_size=3, cxpb=0.7, mutpb=0.2, mut_step=0.05, risk_free_rate=0.0):
        self.pop_size = pop_size
        self.ngen = ngen
        self.tourn_size = tourn_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.mut_step = mut_step
        self.risk_free_rate = risk_free_rate
    
    def random_portfolio(self, num_stocks):
        w = [random.random() for _ in range(num_stocks)]
        total = sum(w)
        return [x/total for x in w]
    
    def normalize(self, ind):
        s = sum(ind)
        if s > 0:
            for i in range(len(ind)):
                ind[i] /= s
        else:
            # If sum is zero (extreme case), evenly distribute weights
            for i in range(len(ind)):
                ind[i] = 1.0/len(ind)
    
    def evaluate(self, individual, expected_returns, cov_matrix):
        # Portfolio return
        w = np.array(individual)
        port_return = w.dot(expected_returns)
        
        # Portfolio variance and std dev using covariance matrix: w^T * Cov * w
        port_variance = w.dot(cov_matrix).dot(w)
        port_vol = np.sqrt(port_variance) if port_variance > 0 else 0.000001

        sharpe = (port_return - self.risk_free_rate) / port_vol
        return sharpe
    
    def tournament_selection(self, pop, fits, tourn_size):
        chosen = random.sample(range(len(pop)), tourn_size)
        best = chosen[0]
        for c in chosen[1:]:
            if fits[c] > fits[best]:
                best = c
        return pop[best][:]
    
    def crossover(self, ind1, ind2):
        size = len(ind1)
        for i in range(size):
            alpha = random.random()
            w1, w2 = ind1[i], ind2[i]
            new_w1 = alpha * w1 + (1 - alpha) * w2
            new_w2 = alpha * w2 + (1 - alpha) * w1
            ind1[i], ind2[i] = new_w1, new_w2
        self.normalize(ind1)
        self.normalize(ind2)
    
    def mutate(self, ind):
        for i in range(len(ind)):
            if random.random() < self.mutpb:
                ind[i] += random.uniform(-self.mut_step, self.mut_step)
        # Remove negatives and normalize
        ind[:] = [max(0, w) for w in ind]
        self.normalize(ind)
    
    def run(self, expected_returns, cov_matrix, symbols=None):

        if not symbols:
            symbols = cov_matrix.columns
        
        num_stocks = len(expected_returns)
        population = [self.random_portfolio(num_stocks) for _ in range(self.pop_size)]
        
        best_individual = None
        best_fitness = float('-inf')
        best_fits_over_time = []
        
        for _ in range(self.ngen):
            # Evaluate population
            fitnesses = [self.evaluate(ind, expected_returns, cov_matrix) for ind in population]

            # Track best of this generation
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx][:]

            best_fits_over_time.append(best_fitness)
            
            # Selection and reproduction
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = self.tournament_selection(population, fitnesses, self.tourn_size)
                parent2 = self.tournament_selection(population, fitnesses, self.tourn_size)
                
                if random.random() < self.cxpb:
                    self.crossover(parent1, parent2)
                
                self.mutate(parent1)
                self.mutate(parent2)
                
                new_population.append(parent1)
                new_population.append(parent2)
            
            population = new_population
            
        fitnesses = [self.evaluate(ind, expected_returns, cov_matrix) for ind in population]
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