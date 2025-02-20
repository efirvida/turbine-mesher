import time

import numpy as np
from matplotlib import pyplot as plt

from turbine_mesher.fem import FemModel, FemModelPETSc
from turbine_mesher.mesh import SquareShapeMesh

"""
Sample problem:
https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html
"""

Solvers = [FemModel, FemModelPETSc]

E = 210e3
NU = 0.3
LOAD = 0, 0, 50

WIDTH, HEIGHT = 1, 1
CELLS_PER_UNIT = 50

NX, NY = WIDTH * CELLS_PER_UNIT, HEIGHT * CELLS_PER_UNIT
mesh = SquareShapeMesh.create_rectangle(
    WIDTH, HEIGHT, NX, NY, quadratic=False, triangular=False
)
# pamesh.view()
K_shape = (6 * mesh.num_nodes, 6 * mesh.num_nodes)

left_nodes = np.array(mesh.node_sets["left"])
right_nodes = np.array(mesh.node_sets["right"])
top_nodes = np.array(mesh.node_sets["top"])
bottom_nodes = np.array(mesh.node_sets["bottom"])

solver_names = []
solver_times = []

for Solver in Solvers:
    start = time.time()
    fem = Solver(mesh, E, NU, 1, use_shell_element=True)
    fem.apply_volumetric_load(LOAD)
    fem.apply_dirichlet_bc(left_nodes, 0)
    fem.apply_dirichlet_bc(right_nodes, 0)
    fem.apply_dirichlet_bc(top_nodes, 0)
    fem.apply_dirichlet_bc(bottom_nodes, 0)
    fem.solve_linear_system()
    end = time.time()
    ellapsed_time = end - start

    solver_names.append(Solver.__name__)
    solver_times.append(ellapsed_time)
    max_disp = fem.u.max()
    print(max_disp)
    fem.write_results(f"{Solver.__name__}.vtk")

plt.figure(figsize=(8, 6))
plt.bar(solver_names, solver_times, color="skyblue")
plt.xlabel("Solver")
plt.ylabel("Tiempo de ejecución (s) ")
plt.title(f"Comparación de tiempos de ejecución de solvers para K {K_shape}")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
