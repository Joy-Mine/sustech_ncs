import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

class NCS:
    def __init__(self, objective_function, dimensions, pop_size, sigma, r, epoch, T_max, w, scope=None, plot=False):
        self.objective_function_individual = objective_function
        self.dimensions = dimensions
        self.pop_size = pop_size
        self.sigma = sigma
        self.r = r
        self.epoch = epoch
        self.T_max = T_max
        self.scope=scope
        self.w = w
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
    def calculate_correlations(self, population, sigma, w):
        correlations = np.zeros(len(population))
        isGlobalSearch = np.zeros(len(population))
        for i, xi in enumerate(population):
            min_distance = np.inf
            sigmai = sigma[i]
            isGlobal = 0
            for j, xj in enumerate(population):
                if i != j:
                    sigmaj = sigma[j]
                    if sigmai > sigmaj * w:
                        isGlobal = 1
                        # print(f'change to global')
                        distance = self.bhattacharyya_distance(xi, sigmai, xj, sigmaj)
                        if distance < min_distance:
                            min_distance = distance
            correlations[i] = min_distance
            # print(f'isglobal is {isGlobal}')
            isGlobalSearch[i] = isGlobal
        return correlations, isGlobalSearch

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

    def update_population(self, population, f_pop, population_prime, f_pop_prime, sigma, lambda_t, count, best_f_solution, w):
        new_population = np.copy(population)
        correlations, isGlobal = self.calculate_correlations(population, sigma, w)
        correlations_prime, isGlobalPrime=self.calculate_correlations(population_prime,sigma, w)

        for i, xi in enumerate(population):
            father_f = (best_f_solution - f_pop[i])
            child_f = (best_f_solution - f_pop_prime[i])
            child_f = child_f / (child_f + father_f + 1e-8)
            corr_p = correlations_prime[i]/(correlations[i]+correlations_prime[i]+1e-8)
            if isGlobalPrime[i] == 0:
                if f_pop_prime[i] < f_pop[i]:
                    new_population[i] = population_prime[i]
                    count[i] += 1
            else:
                if child_f / corr_p < lambda_t:
                    new_population[i] = population_prime[i]
                    count[i] += 1
                else:
                    new_population[i] = population[i]
        return new_population, count, isGlobalPrime
    
    # dimensions, pop_size, sigma, r, epoch, T_max
    def NCS_run(self):
        pool = Pool()
        if self.plot:
            best_f_solutions = []  # 用于存储best_f_solution的历史记录
            t_values = []  # 用于存储对应的t值
        population = self.initialize_population(self.pop_size, self.dimensions, self.scope)
        count = np.full(self.pop_size, 0)
        f_population = self.objective_function(population, pool)
        best_index = np.argmin(f_population)
        best_solution = population[best_index]
        best_f_solution = f_population[best_index]
        # print(population)
        # print(f_population)
        # print(best_f_solution)
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
            # print(f'sigma is {self.sigma}' )
            if new_best_f_solution < best_f_solution:
                best_solution = new_best_solution
                best_f_solution = new_best_f_solution
            population, count, isGlobalPrime = self.update_population(population, f_population, childPopulaiton, f_childPopulation, self.sigma, lambda_t, count, best_f_solution, self.w)
            # print(f'count is {count}')
            # print(f'global is {isGlobalPrime}')
            f_population = self.objective_function(population, pool)  
            if self.plot:
                best_f_solutions.append(best_f_solution)
                t_values.append(t)
            t += 1
        pool.close()
        pool.join()
        if self.plot:
            # 在代码的最后添加以下保存图形的代码
            plt.plot(t_values, best_f_solutions, label='Best f_solution')
            plt.xlabel('t')
            plt.ylabel('Best f_solution')
            plt.title('Best f_solution over time')
            plt.legend()
            plt.savefig('evolution_process.png')  # 将图形保存为PNG文件
            plt.close()  # 关闭图形显示窗口
        return best_solution, best_f_solution

