import numpy as np
from mealpy import FloatVar, Problem, ACOR

class MyProblem(Problem):
    def __init__(self, bounds=None):
        super().__init__(bounds, minmax="min")

    def obj_func(self, solution):
        return np.sum(solution**2)


bounds  = FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta")
problem = MyProblem(bounds=bounds)

model = ACOR.OriginalACOR(epoch=100, pop_size= 50, sample_count= 25, intent_factor= 0.5, zeta = 1.0)
g_best = model.solve(problem)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")