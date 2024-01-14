import numpy as np
from multiprocessing import Pool

class NCS:
    def __init__(self, objective_function, dimensions=30, pop_size=10, sigma=0.2, r=0.8, epoch=10, T_max=30000, scope=None, pre_popu=None, pre_sigma=None):
        self.objective_function_individual = objective_function
        self.dimensions = dimensions
        self.pop_size = pop_size
        self.sigma = sigma
        self.r = r
        self.epoch = epoch
        self.T_max = T_max
        self.scope=scope
        self.pre_popu=pre_popu
        self.pre_sigma=pre_sigma
        # self.pool = Pool()

    def objective_function(self, population, pool):
        results = pool.map(self.objective_function_individual, population)
        return np.array(results)

    def initialize_population(self, pop_size, dimensions, scope):
        if scope is None:
            return np.random.uniform(-100, 100, size=(pop_size, dimensions))
        else:
            population = np.zeros((pop_size, dimensions))
            for j in range(dimensions):
                lower_bound, upper_bound = scope[j]
                population[:, j] = np.random.uniform(lower_bound, upper_bound, size=pop_size)
            return population
        
    # def gaussian_mutation_individual(self, args):
    #     individual, sigma = args
    #     temp = np.random.normal(0, sigma, len(individual))
    #     return individual + temp
    def gaussian_mutation_individual(self, args):
        individual, sigma = args
        temp = np.random.normal(0, sigma, len(individual))
        mutated_individual = individual + temp

        for i in range(self.dimensions):
            lower_bound, upper_bound = self.scope[i]
            length=upper_bound-lower_bound
            if mutated_individual[i] < lower_bound:
                exceed = (lower_bound - mutated_individual[i])%length
                mutated_individual[i] = lower_bound + exceed
            elif mutated_individual[i] > upper_bound:
                exceed = (mutated_individual[i] - upper_bound)%length
                mutated_individual[i] = upper_bound - exceed

        return mutated_individual


    def gaussian_mutation(self, population, sigma, pool):
        p = pool.map(self.gaussian_mutation_individual, zip(population, sigma))
        return np.array(p)

    def bhattacharyya_distance(self, mu_i, sigma_i, mu_j, sigma_j):
        xi_xj = mu_i - mu_j
        big_sigma1 = sigma_i * sigma_i
        big_sigma2 = sigma_j * sigma_j
        big_sigma = (big_sigma1 + big_sigma2) / 2
        small_value = 1e-8
        part1 = 1 / 8 * np.sum(xi_xj *xi_xj / (big_sigma + small_value))
        part2 = (np.sum(np.log(big_sigma + small_value))
                - 1 / 2 * np.sum(np.log(big_sigma1 + small_value))
                - 1 / 2 * np.sum(np.log(big_sigma2 + small_value))
                )
        return part1 + 1 / 2 * part2
    def calculate_correlations(self, population, sigma):
        correlations = np.zeros(len(population))
        for i, xi in enumerate(population):
            min_distance = np.inf
            sigmai = sigma[i]
            for j, xj in enumerate(population):
                if i != j:
                    sigmaj = sigma[j]
                    distance = self.bhattacharyya_distance(xi, sigmai, xj, sigmaj)
                    if distance < min_distance:
                        min_distance = distance
            correlations[i] = min_distance
        return correlations

    def update_sigma(self, pop_size, sigma, count, epoch, r):
        for i in range(pop_size):
            success_rate = count[i] / epoch
            if success_rate > 0.2:
                sigma[i] /= r
            elif success_rate < 0.2:
                sigma[i] *= r
        return sigma

    def generateAndEvalChild(self, population, sigma, pool):
        population_prime = self.gaussian_mutation(population, sigma, pool)
        f_pop_prime = self.objective_function(population_prime, pool)
        return population_prime, f_pop_prime

    def update_best(self, population_prime, f_pop_prime):
        best_index = np.argmin(f_pop_prime)
        best_solution = population_prime[best_index]
        best_f_solution = f_pop_prime[best_index]
        # best_solution = np.copy(best_solution)
        return best_solution, best_f_solution

    def update_population(self, population, f_pop, population_prime, f_pop_prime, sigma, lambda_t, count, best_f_solution):
        new_population = np.copy(population)
        correlations = self.calculate_correlations(population, sigma)
        correlations_prime=self.calculate_correlations(population_prime,sigma)

        for i, xi in enumerate(population):
            father_f = (best_f_solution - f_pop[i])
            child_f = (best_f_solution - f_pop_prime[i])
            child_f = child_f / (child_f + father_f + 1e-8)
            corr_p = correlations_prime[i]/(correlations[i]+correlations_prime[i]+1e-8)
            if child_f / corr_p < lambda_t:
                new_population[i] = population_prime[i]
                count[i]+=1
            else:
                new_population[i] = population[i]
        return new_population, count
    
    # dimensions, pop_size, sigma, r, epoch, T_max
    def NCS_run(self):
        pool = Pool()
        if self.pre_popu is None:
            population = self.initialize_population(self.pop_size, self.dimensions, self.scope)
        else:
            population = self.pre_popu
        if self.pre_sigma is not None:
            self.sigma=self.pre_sigma
        count = np.full(self.pop_size, 0)
        f_population = self.objective_function(population, pool)
        best_index = np.argmin(f_population)
        best_solution = population[best_index]
        best_f_solution = f_population[best_index]
        t = 0
        while t < self.T_max:
            if t % self.epoch == 0 and t > 0:
                self.sigma = self.update_sigma(self.pop_size, self.sigma, count, self.epoch, self.r)
            if t % self.epoch == 0 and t > 0:
                count = np.full(self.pop_size, 0)
            # print(t)
            lambda_t = np.random.normal(1, 0.1 - 0.1 * (t / self.T_max))
            childPopulaiton, f_childPopulation = self.generateAndEvalChild(population, self.sigma, pool) 
            new_best_solution, new_best_f_solution = self.update_best(childPopulaiton, f_childPopulation)
            # print(f'allBest is {best_f_solution}, childBest is {new_best_f_solution}')
            if new_best_f_solution < best_f_solution:
                best_solution = new_best_solution
                best_f_solution = new_best_f_solution
            population, count = self.update_population(population, f_population, childPopulaiton, f_childPopulation, self.sigma, lambda_t, count, best_f_solution)
            f_population = self.objective_function(population, pool)
            
            t += 1
        pool.close()
        pool.join()
        return best_solution, best_f_solution
