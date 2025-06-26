import numpy as np
from mealpy import FloatVar, Problem, ACOR

# Intervalo das variáveis de [-100, 100]
# Quantidade de repetições: 30
# 10 variáveis
# Evaporação de feromônio (ρ): 0.5
# Importância do feromônio (α): 1
# Importância da heurística (β): 2
# a. Configuração 1:
#   i. Tamanho da População: 20
#   ii. Número de Iterações: 500
# b. Configuração 2:
#   i. Tamanho da População: 50
#   ii. Número de Iterações: 1000
# c. Configuração 3:
#   i. Tamanho da População: 100
#   ii. Número de Iterações: 2000

class RotatedEllipticProblem(Problem):
    def __init__(self, bounds=None, angle=np.pi/4):
        """
        Rotated High Conditoned Elliptic Function
        
        Args:
            bounds: Search domain bounds
            angle: Rotation angle in radians (default: pi/4; 45 degrees)
        """
        self.angle = angle
        self.rotation_matrix = None
        super().__init__(bounds, minmax="min")
        self.name = "Rotated High Conditioned Elliptic Function"

    def create_rotation_matrix(self, n):
        """
        Create a rotation matrix for the given dimension n.
        
        Args:
            n: Dimension of the problem
        """
        if n > 2:
            matrix = np.random.randn(n, n)
            q, r = np.linalg.qr(matrix)
            if np.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            return q
        else:
            return np.array([
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)]
            ])
    def obj_func(self, x):
        """
        Calculate the rotated elliptic function value
        
        Args:
            x: numpy array with the solution to evaluate

        Returns:
            float: objective function value
        """
        n = len(x)
        if self.rotation_matrix is None or self.rotation_matrix.shape[0] != n:
            self.rotation_matrix = self.create_rotation_matrix(n)

        x_rotated = np.dot(self.rotation_matrix, x)
        result = 0
        for i in range(n):
            result += (10**6) ** (i / (n - 1)) * x_rotated[i] ** 2
        return result 


class ShiftedRotatedWeierstrassProblem(Problem):
    def __init__(self, bounds=None, shift=None, angle=np.pi/6):
        """
        Shifted and Rotated Weierstrass Function
        
        Args:
            bounds: Search domain bounds
            shift: Shift vector (if None, a random shift vector will be created)
            angle: Rotation angle in radians (default: 30 degrees)
        """
        self.angle = angle
        self.shift = shift
        self.rotation_matrix = None
        super().__init__(bounds, minmax="min")
        self.name = "Shifted and Rotated Weierstrass Function"
    def create_rotation_matrix(self, n):
        """
        Create a rotation matrix for the given dimension n.
        
        Args:
            n: Dimension of the problem
        """
        if n > 2:
            matrix = np.random.randn(n, n)
            q, r = np.linalg.qr(matrix)
            if np.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            return q
        else:
            return np.array([
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)]
            ])
        
    def obj_func(self, x):
        """
        Calculate the shifted rotated Weierstrass function value
        
        Args:
            x: numpy array with the solution to evaluate
        
        Returns:
            float: objective function value
        """
        n = len(x)

        if self.shift is None or len(self.shift) != n:
            self.shift = np.random.uniform(-0.4, 0.4, n)

        if self.rotation_matrix is None or self.rotation_matrix.shape[0] != n:
            self.rotation_matrix = self.create_rotation_matrix(n)

        x_shitfted = x - self.shift
        x_rotated = np.dot(self.rotation_matrix, x_shitfted)

        a = 0.5
        b = 3
        kmax = 20

        sum_x = 0
        for i in range(n):
            for k in range(kmax + 1):
                sum_x += a**k * np.cos(2 * np.pi * b ** k * (x_rotated[i] + 0.5))

        sum_0 = 0
        zero_rotated = np.zeros(n)
        for i in range(n):
            for k in range(kmax + 1):
                sum_0 += a**k * np.cos(2 * np.pi * b ** k * (zero_rotated[i] + 0.5))

        return sum_x - n * sum_0

# Define dimensions and bounds
n_dims = 10  # 10 variables as specified in your comments
bounds_elliptic = FloatVar(lb=(-100.,) * n_dims, ub=(100.,) * n_dims, name="elliptic_vars")
bounds_weierstrass = FloatVar(lb=(-0.5,) * n_dims, ub=(0.5,) * n_dims, name="weierstrass_vars")

# Create problem instances
elliptic_problem = RotatedEllipticProblem(bounds=bounds_elliptic)
weierstrass_problem = ShiftedRotatedWeierstrassProblem(bounds=bounds_weierstrass)

# Run the algorithm for each configuration on each problem
configurations = [
    {"pop_size": 20, "epoch": 500},
    {"pop_size": 50, "epoch": 1000},
    {"pop_size": 100, "epoch": 2000}
]

# Common parameters
common_params = {
    "sample_count": 25,  # Default value
    "intent_factor": 0.5,  # ρ: Pheromone evaporation rate
    "zeta": 1.0  # α: Pheromone importance
}

# Run experiments
for config_id, config in enumerate(configurations, 1):
    print(f"\n=== Configuration {config_id} ===")
    print(f"Population size: {config['pop_size']}, Iterations: {config['epoch']}")
    
    # Solve Rotated Elliptic
    print(f"\nSolving {elliptic_problem.name}...")
    model_elliptic = ACOR.OriginalACOR(
        epoch=config['epoch'], 
        pop_size=config['pop_size'], 
        **common_params
    )
    best_elliptic = model_elliptic.solve(elliptic_problem)
    print(f"Best fitness: {best_elliptic.target.fitness}")
    print(f"Best solution: {best_elliptic.solution[:5]}... (showing first 5 dimensions)")
    
    # Solve Shifted Rotated Weierstrass
    print(f"\nSolving {weierstrass_problem.name}...")
    model_weierstrass = ACOR.OriginalACOR(
        epoch=config['epoch'], 
        pop_size=config['pop_size'], 
        **common_params
    )
    best_weierstrass = model_weierstrass.solve(weierstrass_problem)
    print(f"Best fitness: {best_weierstrass.target.fitness}")
    print(f"Best solution: {best_weierstrass.solution[:5]}... (showing first 5 dimensions)")
    