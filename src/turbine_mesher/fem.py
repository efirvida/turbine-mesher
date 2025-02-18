from abc import ABC, abstractmethod

import meshio
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from turbine_mesher.elements import (
    LagrangeElement,
    QuadElement,
    SerendipityElement,
    TriElement,
    TriQuadElement,
)

# Mapeo del número de nodos al tipo de elemento
ELEMENTS_MAP = {
    3: TriElement,
    4: QuadElement,
    6: TriQuadElement,
    8: SerendipityElement,
    9: LagrangeElement,
}


class FEA(ABC):
    def __init__(self, mesh, E, nu):
        self.mesh = mesh
        self.dim = self.mesh.dim
        self.ndof = self.dim * self.mesh.num_nodes
        self.E = E
        self.nu = nu
        self._u = None
        self._K = None
        self._f = None

    @property
    def u(self):
        return self._u

    @property
    def K(self):
        return self._K

    @property
    def f(self):
        return self._f

    @abstractmethod
    def assemble_K(self): ...

    @abstractmethod
    def apply_volumetric_load(self, f_vec): ...

    @abstractmethod
    def apply_dof_load(self, nodes, values): ...

    @abstractmethod
    def apply_dirichlet_bc(self, nodes, value): ...

    @abstractmethod
    def solve_linear_system(self): ...

    def write_results(self, output_file: str):
        """Escribe los resultados en un archivo VTK."""
        points = self.mesh.nodes
        U = self.u.reshape(-1, self.dim)
        F = self.f.reshape(-1, self.dim)

        cells = []
        for element_kind, labels in self.mesh.elements_class.items():
            if labels:
                cells.append(
                    (
                        element_kind.lower(),  # <- FIXME use better thing here
                        [[i for i in el if i != -1] for el in self.mesh.elements[labels]],
                    )
                )

        point_data = {"U": U, "F": F}
        mesh_object = meshio.Mesh(points, cells, point_data=point_data)
        mesh_object.write(output_file, file_format="vtk")


class FemModel(FEA):
    def __init__(self, mesh, E, nu):
        super().__init__(mesh, E, nu)
        self._K = np.zeros((self.ndof, self.ndof), dtype=np.float32)
        self._f = np.zeros((self.mesh.num_nodes, self.dim))

    def assemble_K(self):
        for element in self.mesh.elements_map.values():
            coords = self.mesh.nodes[element]
            num_nodes = len(element)
            element_type = ELEMENTS_MAP[num_nodes]
            el = element_type(coords, self.E, self.nu)
            Ke = el.Ke
            assert np.allclose(Ke, Ke.T), "La matriz de rigidez elemental no es simétrica"

            nodes = np.array(element)
            global_dofs = np.empty(el.dofs_per_node * num_nodes, dtype=int)
            for i in range(el.dofs_per_node):
                global_dofs[i :: el.dofs_per_node] = el.dofs_per_node * nodes + i

            self._K[np.ix_(global_dofs, global_dofs)] += Ke
        return self._K

    def apply_volumetric_load(self, f_vec):
        """
        Ensambla el vector global de fuerzas integrando la carga en cada elemento.

        Parámetros
        ----------
        f_vec : array, shape (2,)
            Vector de carga volumétrica, por ejemplo [0, -rho * g].

        Retorna
        -------
        f_global : array, shape (ndof,)
        """
        f_global = np.zeros(self.ndof, dtype=float)
        for element in self.mesh.elements_map.values():
            # Se omiten los índices -1 si existen
            coords = self.mesh.nodes[element]

            # Crea el elemento (nota: podrías parametrizar el número de puntos de cuadratura)
            element_type = ELEMENTS_MAP[len(element)]
            el = element_type(coords, self.E, self.nu)
            points, weights = el.integration_points

            Fe = np.zeros(el.dofs)  # Vector de carga elemental

            # Recorre los puntos de integración
            for (xi, eta), w in zip(points, weights):
                # Evalúa las funciones de forma en (xi, eta)
                N = el.shape_functions(xi, eta)  # Debe retornar un array de longitud n_nodos

                # Calcula el Jacobiano como en la asamblea de la matriz de rigidez
                dN_dxi, dN_deta = el.shape_function_derivatives(xi, eta)
                J = np.array(
                    [
                        [dN_dxi @ coords[:, 0], dN_deta @ coords[:, 0]],
                        [dN_dxi @ coords[:, 1], dN_deta @ coords[:, 1]],
                    ]
                )
                detJ = np.linalg.det(J)

                # Contribución elemental: se asume que la distribución es la misma en x e y
                # Se deben asignar las contribuciones a cada DOF:
                # Por ejemplo, si los DOF están ordenados como [u1_x, u1_y, u2_x, u2_y, ...]
                Fe[0::2] += N * f_vec[0] * detJ * w
                Fe[1::2] += N * f_vec[1] * detJ * w

            # Mapear los DOF locales a los globales
            nodes = np.array(element)
            global_dofs = np.empty(el.dofs_per_node * len(element), dtype=int)
            for i in range(el.dofs_per_node):
                global_dofs[i :: el.dofs_per_node] = el.dofs_per_node * nodes + i

            # Sumar la contribución elemental al vector global
            f_global[global_dofs] += Fe

        self._f = f_global

    def apply_dof_load(self, nodes, values):
        """Asigna la carga (vector de fuerza) en los nodos indicados."""
        self._f[nodes] = values

    def apply_dirichlet_bc(self, nodes, value):
        """
        Aplica condiciones de contorno de Dirichlet (desplazamientos fijos)
        en los nodos indicados y ajusta el vector de fuerza para que
        no se aplique carga en esos grados de libertad.
        """
        for i in nodes:
            for d in range(self.dim):
                idx = self.dim * i + d
                self._K[idx, :] = value
                self._K[:, idx] = value
                self._K[idx, idx] = 1

                self.f[idx] = 0.0

    def solve_linear_system(self):
        self._u = np.linalg.solve(self._K, self.f)
        return self._u


class FemModelPETSc(FEA):
    """
    Finite Element Model using PETSc for linear finite element analysis.

    This model assembles the global stiffness matrix, mass matrix, and load vector
    from element-level contributions. Each element is assumed to provide its consistent
    stiffness matrix (Ke), mass matrix (Me), and load vector (via load_vector method)
    based on an isoparametric formulation.

    The element formulations are:

        Stiffness matrix:
            .. math::
                K_e = \\int_{\\Omega} B^T C B \\; d\\Omega,

        Consistent mass matrix:
            .. math::
                M_e = \\int_{\\Omega} \\rho N^T N \\; d\\Omega,

        Load vector (body force):
            .. math::
                f_e = \\int_{\\Omega} N^T \\mathbf{b} \\; d\\Omega,

    where:
        - \\(B\\) is the strain-displacement matrix,
        - \\(C\\) is the constitutive (elasticity) matrix,
        - \\(N\\) is the shape function matrix,
        - \\(\\rho\\) is the material density,
        - \\(\\mathbf{b}\\) is the body force vector,
        - \\(\\Omega\\) is the element domain.
    """

    def __init__(self, mesh, E, nu):
        """
        Initialize the FEM model with the provided mesh and material properties.

        Parameters
        ----------
        mesh : MeshType
            The mesh object containing nodes and elements.
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        """
        super().__init__(mesh, E, nu)
        # Create PETSc matrices and vectors for global system assembly.
        self._K = PETSc.Mat().createAIJ(size=(self.ndof, self.ndof), nnz=100)
        self._M = PETSc.Mat().createAIJ(size=(self.ndof, self.ndof), nnz=100)
        self._f = PETSc.Vec().createWithArray(np.zeros(self.ndof))
        self._u = PETSc.Vec().createWithArray(np.zeros(self.ndof))
        # Assuming a 2D problem.
        self.dim = 2

    @property
    def u(self):
        """
        Get the computed displacement vector.

        Returns
        -------
        np.ndarray
            Displacement vector as a NumPy array.
        """
        return self._u.getArray()

    @property
    def K(self):
        """
        Get the assembled global stiffness matrix.

        Returns
        -------
        np.ndarray
            Global stiffness matrix as a NumPy array.
        """
        return self._K.getArray()

    @property
    def M(self):
        """
        Get the assembled global mass matrix.

        Returns
        -------
        np.ndarray
            Global mass matrix as a NumPy array.
        """
        return self._M.getArray()

    @property
    def f(self):
        """
        Get the assembled global load vector.

        Returns
        -------
        np.ndarray
            Global load vector as a NumPy array.
        """
        return self._f.getArray()

    @staticmethod
    def _compute_global_dofs(nodes: np.ndarray, dofs_per_node: int) -> np.ndarray:
        """
        Compute the global degrees of freedom indices for a given set of node indices.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node indices.
        dofs_per_node : int
            Number of degrees of freedom per node.

        Returns
        -------
        np.ndarray
            Array of global degrees of freedom indices.
        """
        # Vectorized implementation:
        return (
            np.repeat(nodes, dofs_per_node) * dofs_per_node
            + np.tile(np.arange(dofs_per_node), nodes.size)
        ).astype(np.int32)

    def assemble_K(self):
        """
        Assemble the global stiffness matrix from element stiffness matrices.

        The global stiffness matrix is assembled as:

            .. math::
                K = \\sum_{e} T_e^T K_e T_e,

        where \\(K_e\\) is the stiffness matrix of element \\(e\\) and \\(T_e\\) is the transformation
        from element degrees of freedom to global degrees of freedom.

        Returns
        -------
        PETSc.Mat
            The assembled global stiffness matrix.
        """
        for element in self.mesh.elements_map.values():
            coords = self.mesh.nodes[element]

            element_type = ELEMENTS_MAP[len(element)]
            el = element_type(coords, self.E, self.nu)
            Ke = el.Ke
            assert np.allclose(Ke, Ke.T), "Element stiffness matrix is not symmetric."

            nodes = np.array(element)
            global_dofs = FemModelPETSc._compute_global_dofs(nodes, el.dofs_per_node)

            self._K.setValues(global_dofs, global_dofs, Ke, addv=PETSc.InsertMode.ADD)

        self._K.assemble()
        return self._K

    def assemble_M(self):
        """
        Assemble the global mass matrix from element mass matrices.

        The global mass matrix is assembled as:

            .. math::
                M = \\sum_{e} T_e^T M_e T_e,

        where \\(M_e\\) is the mass matrix of element \\(e\\) and \\(T_e\\) is the transformation
        from element degrees of freedom to global degrees of freedom.

        Returns
        -------
        PETSc.Mat
            The assembled global mass matrix.
        """
        for element in self.mesh.elements_map.values():
            coords = self.mesh.nodes[element]

            element_type = ELEMENTS_MAP[len(element)]
            el = element_type(coords, self.E, self.nu)
            Me = el.Me.astype(np.float32)

            nodes = np.array(element)
            global_dofs = FemModelPETSc._compute_global_dofs(nodes, el.dofs_per_node)

            self._M.setValues(global_dofs, global_dofs, Me, addv=PETSc.InsertMode.ADD)

        self._M.assemble()
        return self._M

    def apply_volumetric_load(self, f_vec):
        """
        Assemble the global load vector from a constant volumetric body force.

        The element load vector is computed using the element's own load_vector method,
        which evaluates:

            .. math::
                f_e = \\int_{\\Omega} N^T \\mathbf{b} \\; d\\Omega,

        where \\(N\\) is the shape function matrix and \\(\\mathbf{b}\\) is the body force vector.

        Parameters
        ----------
        f_vec : np.ndarray
            A 2-element array representing the body force (force per unit area).

        Returns
        -------
        None
        """
        f_global = PETSc.Vec().createWithArray(np.zeros(self.ndof))
        for element in self.mesh.elements_map.values():
            coords = self.mesh.nodes[element]

            element_type = ELEMENTS_MAP[len(element)]
            el = element_type(coords, self.E, self.nu)
            Fe = el.load_vector(f_vec).astype(np.float32)

            nodes = np.array(element)
            global_dofs = FemModelPETSc._compute_global_dofs(nodes, el.dofs_per_node)

            f_global.setValues(global_dofs, Fe, addv=PETSc.InsertMode.ADD)

        f_global.assemble()
        self._f = f_global

    def apply_dof_load(self, nodes, values):
        """
        Apply prescribed loads directly at specified degrees of freedom.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node indices where the loads are applied.
        values : np.ndarray
            Array of load values corresponding to each degree of freedom at the specified nodes.

        Returns
        -------
        None
        """
        dofs = FemModelPETSc._compute_global_dofs(nodes, self.dim)
        self._f.setValues(dofs, np.tile(values, nodes.size))

    def apply_dirichlet_bc(self, nodes, value):
        """
        Apply Dirichlet boundary conditions (prescribed displacements) at specified nodes.

        This method modifies the global stiffness matrix and load vector to enforce the specified displacement.
        It now supports nonzero Dirichlet conditions by adjusting the load vector accordingly.

        Parameters
        ----------
        nodes : np.ndarray
            Array of node indices where the Dirichlet boundary conditions are applied.
        value : float
            Prescribed displacement value.

        Returns
        -------
        None
        """
        for i in nodes:
            for d in range(self.dim):
                idx = self.dim * i + d
                # Adjust the global load vector and zero out the row/column corresponding to the fixed DOF.
                self._K.zeroRowsColumns([idx], diag=1.0, x=self._f)
                self._f.setValue(idx, value)

    def solve_linear_system(self):
        """
        Solve the linear system using PETSc solvers with optimal options for symmetric positive definite systems.

        The solver uses the Conjugate Gradient (CG) method with an Incomplete Cholesky (GAMG) preconditioner,
        and sets appropriate tolerances for convergence.

        Returns
        -------
        np.ndarray
            The computed displacement vector.
        """
        ksp = PETSc.KSP().create()
        ksp.setType("cg")
        pc = ksp.getPC()
        pc.setType("gamg")
        ksp.setTolerances(rtol=1e-8, atol=1e-12)
        ksp.setFromOptions()

        ksp.setOperators(self._K)
        ksp.solve(self._f, self._u)
        return self._u.getArray()

    def solve_modal_analysis(self, num_modes=6):
        """
        Performs modal analysis to determine the system's natural frequencies and vibration modes.

        Parameters:
        -----------
        num_modes : int, optional
            Number of eigenmodes to compute (default is 6).

        Returns:
        --------
        eigvals : np.ndarray
            Array containing the natural frequencies in radians per second.
        eigvecs : np.ndarray
            Array containing the corresponding vibration modes.
        """
        eps = SLEPc.EPS().create()
        eps.setOperators(self._K, self._M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
        eps.setDimensions(num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.solve()

        nconv = eps.getConverged()
        if nconv == 0:
            raise RuntimeError("No eigenvalues were found.")

        eigvals = []
        eigvecs = []
        vr, vi = self._K.getVecs()
        for i in range(min(nconv, num_modes)):
            eigval = eps.getEigenpair(i, vr, vi)
            eigvals.append(np.sqrt(eigval))
            eigvecs.append(vr.getArray())

        return np.array(eigvals), np.array(eigvecs)
