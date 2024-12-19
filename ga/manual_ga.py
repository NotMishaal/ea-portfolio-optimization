import random
import numpy as np

########################
# GA Parameter Settings
########################
POP_SIZE = 50        # Number of individuals in the population
NGEN = 20            # Number of generations
TOURN_SIZE = 3       # Tournament size for selection
CXPB = 0.7           # Crossover probability
MUTPB = 0.2          # Mutation probability
MUT_STEP = 0.05      # Mutation step size
RISK_FREE_RATE = 0.0 # Risk-free rate assumed for Sharpe ratio

def random_portfolio(num_stocks):
    """Create a random portfolio (individual) with non-negative weights summing to 1."""
    w = [random.random() for _ in range(num_stocks)]
    total = sum(w)
    return [x/total for x in w]

def evaluate(individual, expected_returns, expected_vol):
    """Compute the fitness (Sharpe ratio) of an individual."""
    w = individual
    port_return = sum(w[i] * expected_returns[i] for i in range(len(w)))
    port_vol = sum(w[i] * expected_vol[i] for i in range(len(w)))
    if port_vol == 0:
        # Heavily penalize zero-vol (degenerate) solutions
        return -9999
    sharpe = (port_return - RISK_FREE_RATE) / port_vol
    return sharpe

def tournament_selection(pop, fits, tourn_size):
    """Select one individual from the population using tournament selection."""
    # Randomly choose 'tourn_size' individuals and pick the best
    chosen = random.sample(range(len(pop)), tourn_size)
    best = chosen[0]
    for c in chosen[1:]:
        if fits[c] > fits[best]:
            best = c
    return pop[best][:]  # return a copy

def crossover(ind1, ind2):
    """Blend crossover: For each gene, mix the values and re-normalize."""
    size = len(ind1)
    for i in range(size):
        alpha = random.random()
        w1, w2 = ind1[i], ind2[i]
        new_w1 = alpha * w1 + (1 - alpha) * w2
        new_w2 = alpha * w2 + (1 - alpha) * w1
        ind1[i], ind2[i] = new_w1, new_w2
    # Normalize
    normalize(ind1)
    normalize(ind2)

def mutate(ind):
    """Mutate an individual's weights by adding small random changes and re-normalizing."""
    for i in range(len(ind)):
        if random.random() < MUTPB:
            ind[i] += random.uniform(-MUT_STEP, MUT_STEP)
    # Remove negatives and normalize again
    ind[:] = [max(0, w) for w in ind]
    normalize(ind)

def normalize(ind):
    """Ensure weights sum to 1."""
    s = sum(ind)
    if s > 0:
        for i in range(len(ind)):
            ind[i] /= s
    else:
        # If all are zero, distribute evenly
        n = len(ind)
        for i in range(n):
            ind[i] = 1.0 / n

def run_ga(expected_returns, expected_vol):
    """Run the genetic algorithm and return the best individual found."""
    num_stocks = len(expected_returns)
    
    # Initialize population
    population = [random_portfolio(num_stocks) for _ in range(POP_SIZE)]
    
    # Track the best solution
    best_individual = None
    best_fitness = float('-inf')
    best_fits_over_time = []
    
    for gen in range(NGEN):
        # Evaluate the population
        fitnesses = [evaluate(ind, expected_returns, expected_vol) for ind in population]
        
        # Track the best in this generation
        gen_best_idx = np.argmax(fitnesses)
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx][:]

        best_fits_over_time.append(best_fitness)
        
        # Selection: Create a new generation
        new_population = []
        for _ in range(POP_SIZE // 2):
            # Select parents
            parent1 = tournament_selection(population, fitnesses, TOURN_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURN_SIZE)
            
            # Crossover
            if random.random() < CXPB:
                crossover(parent1, parent2)
            
            # Mutation
            mutate(parent1)
            mutate(parent2)
            
            new_population.append(parent1)
            new_population.append(parent2)
        
        population = new_population
    
    # Final evaluation to confirm best
    fitnesses = [evaluate(ind, expected_returns, expected_vol) for ind in population]
    final_best_idx = np.argmax(fitnesses)
    if fitnesses[final_best_idx] > best_fitness:
        best_fitness = fitnesses[final_best_idx]
        best_individual = population[final_best_idx][:]

    return best_individual, best_fitness, best_fits_over_time
