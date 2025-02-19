import time

import numpy as np

from turbine_mesher.fem import FemModel, FemModelPETSc
from turbine_mesher.mesh import SquareShapeMesh

E = 4000000
NU = 0.3
RHO = 3000

WIDTH, HEIGHT = 0.1, 1

NX, NY = 5, 30
mesh = SquareShapeMesh.create_rectangle(
    WIDTH, HEIGHT, NX, NY, quadratic=False, triangular=False
)

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
