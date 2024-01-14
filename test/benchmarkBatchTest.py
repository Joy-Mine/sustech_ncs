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

    # F6
    best=390
    scope = np.array([[-100, 100]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F7
    best=-180
    scope = np.array([[-1000, 1000]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F8
    best=-140
    scope = np.array([[-32, 32]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F9
    best=-330
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F10
    best=-330
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F11
    best=90
    scope = np.array([[-0.5, 0.5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F12
    best=-460
    scope = np.array([[-np.pi, np.pi]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F13
    best=-130
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F14
    best=-300
    scope = np.array([[-100, 100]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F15
    best=120
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F16
    best=120
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F17
    best=120
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F18
    best=10
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F19
    best=10
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F20
    best=10
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F21
    best=360
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F22
    best=360
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F23
    best=360
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    
    # F24
    best=260
    scope = np.array([[-5, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))

    # F25
    best=260
    scope = np.array([[2, 5]] * dimension0)
    function=benchmark.F62005(ndim=dimension0).evaluate

    f_solutions=np.array([])
    for i in range(5):
        optimizer = NCS(function, dimensions=dimension0, pop_size=pop_size0, sigma=np.full(pop_size0, sigma0), r=r0, epoch=epoch0, T_max=T0 ,scope=scope, plot=False)
        best_solution, best_f_solution = optimizer.NCS_run()
        f_solutions=np.append(f_solutions, best_f_solution)
        # print("Best solution found by NCS:", best_solution)
        # print("Objective function value:", best_f_solution)
    print(f"Function: {optimizer.objective_function_individual}, \n"
        f"Dimensions: {optimizer.dimensions}, Population Size: {optimizer.pop_size}, Sigma: {sigma0}, \n"
        f"R: {optimizer.r}, Epoch: {optimizer.epoch}, T_max: {optimizer.T_max}, "
        f"Scope: {optimizer.scope[0]}")
    print("--------------------")
    print("历次最优评估值：", f_solutions)
    print("--------------------")
    print("均值：", np.mean(f_solutions)-best)
    print("标准差：", np.std(f_solutions))
    