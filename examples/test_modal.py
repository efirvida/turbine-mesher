import os
import time

import numpy as np

from turbine_mesher.fem import FemModel, FemModelPETSc
from turbine_mesher.mesh import SquareShapeMesh

"""
Sample problem:
https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html
"""
os.environ["QT_QPA_PLATFORM"] = "xcb"


E = 4000000
NU = 0.3
RHO = 3000

WIDTH, HEIGHT = 0.1, 1

NX, NY = 5, 30
mesh = SquareShapeMesh.create_rectangle(
    WIDTH, HEIGHT, NX, NY, quadratic=False, triangular=False
)
# mesh.view()
# mesh.write_mesh("beam.inp")
fixed_nodes = np.array(mesh.node_sets["bottom"])

start = time.time()
fem_np = FemModel(mesh, E, NU, RHO)
fem_np.assemble_K()
fem_np.assemble_M()
fem_np.apply_dirichlet_bc(fixed_nodes, 0.0)
n_freqs, _ = fem_np.solve_modal_analysis(10)

start = time.time()
fem_petsc = FemModelPETSc(mesh, E, NU, RHO)
fem_petsc.assemble_K()
fem_petsc.assemble_M()
fem_petsc.apply_dirichlet_bc(fixed_nodes, 0.0)
p_freqs, _ = fem_petsc.solve_modal_analysis(10)

print(
    f"Numpy Solution:\nFreq: {'\n'.join(f'Mode {mode}: freq {f:2f}' for mode, f in enumerate(n_freqs))}"
)
print(
    f"PETSc Solution:\nFreq: {'\n'.join(f'Mode {mode}: freq {f:2f}' for mode, f in enumerate(p_freqs))}"
)

# # print("Primer cuadrante 6x6 de la matriz A:")
# print(np.array2string(fem_np.K[:8, :8], formatter={"float_kind": "{:.0e}".format}))

# # print("\nPrimer cuadrante 6x6 de la matriz M:")
# print(np.array2string(fem_np.M[:8, :8], formatter={"float_kind": "{:.2f}".format}))

print(np.all(fem_np.K == fem_petsc.K))
print(np.all(fem_np.M == fem_petsc.M))
