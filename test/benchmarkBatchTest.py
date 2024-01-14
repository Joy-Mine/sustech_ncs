import numpy as np

# import optproblems.cec2005 as benchmark
import opfunu.cec_based.cec2005 as benchmark
from sustech_ncs import NCS


if __name__=='__main__':

    dimension0=30
    pop_size0=10 #
    sigma0=0.2  # 注意sigma不能是整数
    r0=0.80      #
    epoch0=10   #
    T0=30
    scope = np.array([[-np.pi, np.pi]] * dimension0)

    function=benchmark.F62005(ndim=dimension0).evaluate # opfunu
    # function=benchmark.F1(num_variables=dimension0) # optproblems

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        print("Best solution found by NCS:", best_solution)
        print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print(f_solutions)
    print("--------------------")
    print(np.mean(f_solutions))
    print(np.std(f_solutions))