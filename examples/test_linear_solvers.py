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
LOAD = -(2e-3 * 9.81)

WIDTH, HEIGHT = 10, 1
CELLS_PER_UNIT = 10

NX, NY = WIDTH * CELLS_PER_UNIT, HEIGHT * CELLS_PER_UNIT
mesh = SquareShapeMesh.create_rectangle(
    WIDTH, HEIGHT, NX, NY, quadratic=True, triangular=False
)
# mesh.view()
K_shape = (2 * mesh.num_nodes, 2 * mesh.num_nodes)

left_nodes = np.array(mesh.node_sets["left"])
right_nodes = np.array(mesh.node_sets["right"])

solver_names = []
solver_times = []

for Solver in Solvers:
    start = time.time()
    fem = Solver(mesh, E, NU, 1)
    fem.assemble_K()
    fem.assemble_M()
    fem.apply_volumetric_load([0, LOAD])
    fem.apply_dirichlet_bc(left_nodes, 0)
    fem.apply_dirichlet_bc(right_nodes, 0)
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


CurrentSolver = Solvers[-1]

max_displacements = []
num_elements = []
for CELLS_PER_UNIT in [10, 15, 20, 25, 30, 35, 40]:
    NX, NY = WIDTH * CELLS_PER_UNIT, HEIGHT * CELLS_PER_UNIT
    mesh = SquareShapeMesh.create_rectangle(
        WIDTH, HEIGHT, NX, NY, quadratic=True, triangular=False
    )
    num_elements.append(len(mesh.elements))

    # Condiciones de frontera
    left_nodes = np.array(mesh.node_sets["left"])
    right_nodes = np.array(mesh.node_sets["right"])

    # Resolver FEM
    start_time = time.time()
    fem_model = CurrentSolver(mesh, E, NU)
    fem_model.assemble_K()
    fem_model.assemble_M()
    fem_model.apply_volumetric_load([0, LOAD])
    fem_model.apply_dirichlet_bc(left_nodes, 0)
    fem_model.apply_dirichlet_bc(right_nodes, 0)
    fem_model.solve_linear_system()
    end_time = time.time()

    # Obtener desplazamiento máximo
    max_disp = fem_model.u.max()
    max_displacements.append(max_disp)

    print(
        f"Mesh {NX}x{NY}: max displacement = {max_disp:.3e}, time = {end_time - start_time:.3f} s"
    )

# Graficar convergencia
plt.figure(figsize=(8, 5))
plt.loglog(
    num_elements,
    max_displacements,
    marker="o",
    linestyle="-",
    label="Máxima deformación",
)
plt.xlabel("Número de elementos")
plt.ylabel("Máxima deformación")
plt.title("Convergencia de malla basado en la máxima deformación")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
