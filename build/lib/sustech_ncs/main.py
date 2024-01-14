import numpy as np
# from NCS_noMul_pack import NCS
from NCS_mul_pack import NCS
# from functions import custom_objective_function
# import optproblems.cec2005 as benchmark
import opfunu.cec_based.cec2005 as benchmark

if __name__=='__main__':

    dimension0=30
    pop_size0=30 #
    sigma0=2.0  # 注意sigma不能是整数
    r0=0.90      #
    epoch0=40   #
    T0=4000
    scope = np.array([[-np.pi, np.pi]] * dimension0)

    function=benchmark.F122005(ndim=dimension0).evaluate # opfunu
    # function=benchmark.F1(num_variables=dimension0) # optproblems
    optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope)

    best_solution, best_f_solution = optimizer.NCS_run()
    print("Best solution found by NCS:", best_solution)
    print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, Dimensions: {optimizer.dimensions}, \n"
        f"Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")

    # # opfunu用法
    # dimension = 10
    # problem = benchmark.F12005(dimension)
    # solution = [0.1] * dimension 
    # value = problem.evaluate(solution)
    # print(value)

    # # optproblems用法
    # dimension = 10
    # problem = benchmark.F1(dimension)
    # solution = [0.1] * dimension
    # value = problem(solution)