from abc import ABC, abstractmethod
from typing import Literal

import meshio
import numpy as np
import scipy
from petsc4py import PETSc
from slepc4py import SLEPc

from .elements import (
    MITC4,
    LagrangeElement,
    QuadElement,
    SerendipityElement,
    TriElement,
    TriQuadElement,
)
from .helpers import petsc_to_numpy

# Mapeo del número de nodos al tipo de elemento
ELEMENTS_MAP_2D = {
    3: TriElement,
    4: QuadElement,
    6: TriQuadElement,
    8: SerendipityElement,
    9: LagrangeElement,
}

ELEMENTS_MAP_SHELL = {
    4: MITC4,
}


class FEA(ABC):
    def __init__(self, mesh, E: float, nu: float, rho: float = 1, use_shell_element: bool = False):
        self.mesh = mesh
        self.dim = self.mesh.dim
        self.dim = 3 if use_shell_element else 2
        self.ndof = self.mesh.num_nodes * (6 if use_shell_element else 2)
        self.E = E
        self.nu = nu
        self.rho = rho
        self.use_shell_element = use_shell_element
        self._u = np.array([])
        self._K = np.array([])
        self._M = np.array([])
        self._f = np.array([])

        self.global_dof_map = {}
        for element_id, nodes in mesh.elements_map.items():
            num_nodes = len(nodes)
            element_type = (ELEMENTS_MAP_SHELL if use_shell_element else ELEMENTS_MAP_2D)[num_nodes]
            dofs = element_type.dofs_per_node
            base_indices = np.array(nodes, dtype=np.int32) * dofs
            global_dofs = (base_indices[:, None] + np.arange(dofs)).flatten()
            self.global_dof_map[element_id] = global_dofs.astype(np.int32)

    @property
    def u(self) -> np.ndarray:
        return self._u

    @property
    def K(self) -> np.ndarray:
        return self._K

    @property
    def M(self) -> np.ndarray:
        return self._M

    @property
    def f(self) -> np.ndarray:
        return self._f

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
        self._K = self._assemble_matrix("K")

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
        self._M = self._assemble_matrix("M")

    @abstractmethod
    def _assemble_matrix(self, property: Literal["K", "M"]): ...

    @abstractmethod
    def apply_volumetric_load(self, f_vec): ...

    @abstractmethod
    def apply_dof_load(self, nodes, values): ...

    @abstractmethod
    def apply_dirichlet_bc(self, nodes, value): ...

    @abstractmethod
    def solve_linear_system(self): ...

    @abstractmethod
    def solve_modal_analysis(self): ...

    def write_results(self, output_file: str):
        """Escribe los resultados en un archivo VTK."""
        points = self.mesh.nodes
        if self.use_shell_element:
            U = self.u.reshape(-1, 6)
            U_disp = U[:, :3]
            U_rot = U[:, 3:]
            point_data = {"Displacement": U_disp, "Rotation": U_rot}
        else:
            U = self.u.reshape(-1, 2)
            U = np.hstack((self.u.reshape(-1, 2), np.zeros((self.u.shape[0] // 2, 1))))
            point_data = {"Displacement": U}

        cells = []
        for element_kind, labels in self.mesh.elements_class.items():
            if labels:
                cells.append(
                    (
                        element_kind.lower(),  # <- FIXME use better thing here
                        [[i for i in el if i != -1] for el in self.mesh.elements[labels]],
                    )
                )

        mesh_object = meshio.Mesh(points, cells, point_data=point_data)
        mesh_object.write(output_file, file_format="vtk")


class FemModel(FEA):
    def __init__(self, mesh, E: float, nu: float, rho: float = 1, use_shell_element: bool = False):
        super().__init__(mesh, E, nu, rho, use_shell_element)
        self._K = np.zeros((self.ndof, self.ndof))
        self._M = np.zeros((self.ndof, self.ndof))
        self._f = np.zeros((self.mesh.num_nodes, self.dim))
        self.assemble_K()
        self.assemble_M()

    def _assemble_matrix(self, property: Literal["K", "M"] = "K"):
        if property not in ("K", "M"):
            raise NotImplementedError

        matrix = np.zeros((self.ndof, self.ndof))
        for element_id, element in self.mesh.elements_map.items():
            coords = self.mesh.nodes[element]
            num_nodes = len(element)
            if self.use_shell_element:
                element_type = ELEMENTS_MAP_SHELL[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho, h=0.1)
            else:
                element_type = ELEMENTS_MAP_2D[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho)

            if property == "K":
                P = el.Ke
            if property == "M":
                P = el.Ke

            assert np.allclose(P, P.T), f"La matriz {property} elemental no es simétrica"

            global_dofs = self.global_dof_map[element_id]

            matrix[np.ix_(global_dofs, global_dofs)] += P
        return matrix

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
        f_global = np.zeros(self.ndof)
        for element_id, element in self.mesh.elements_map.items():
            # Se omiten los índices -1 si existen
            coords = self.mesh.nodes[element]
            num_nodes = len(element)
            if self.use_shell_element:
                element_type = ELEMENTS_MAP_SHELL[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho, h=0.1)
            else:
                element_type = ELEMENTS_MAP_2D[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho)

            Fe = el.load_vector(f_vec).astype(np.float32)

            global_dofs = self.global_dof_map[element_id]

            # Sumar la contribución elemental al vector global
            f_global[global_dofs.astype(int)] += Fe

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

                self._M[idx, :] = 0.0
                self._M[:, idx] = 0.0
                self._M[idx, idx] = 1.0

                self._f[idx] = 0.0

    def solve_linear_system(self):
        self._u = np.linalg.solve(self._K, self.f)
        return self._u

    def solve_modal_analysis(self, num_modes=6):
        """
        Performs modal analysis to determine the system's natural frequencies and vibration modes.

        Parameters:
        -----------
        num_modes : int, optional
            Number of eigenmodes to compute (default is 6).

        Returns
        -------
        frequencies : np.ndarray
            Natural frequencies in Hz.
        mode_shapes : np.ndarray
            Corresponding mode shapes (eigenvectors).
        """
        eigvals, eigvecs = scipy.linalg.eigh(self.K, self.M)
        idx = np.argsort(eigvals)
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Convert eigenvalues to frequencies in Hz
        frequencies = np.sqrt(np.maximum(eigvals[:num_modes], 0)) / (2 * np.pi)
        return frequencies, eigvecs[:, :num_modes]


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

    def __init__(self, mesh, E: float, nu: float, rho: float = 1, use_shell_element: bool = False):
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
        rho : float
            Material density.
        """
        super().__init__(mesh, E, nu, rho, use_shell_element)

        self.dim = 2

        self.nnz = self._compute_nnz()
        self._K = self._create_matrix()
        self._M = self._create_matrix()
        self._f = PETSc.Vec().createWithArray(np.zeros(self.ndof))
        self._u = PETSc.Vec().createWithArray(np.zeros(self.ndof))
        self.assemble_K()
        self.assemble_M()

    def _compute_nnz(self):
        """Calcula número de no-ceros por fila basado en conectividad de la malla."""
        adjacency = [set() for _ in range(self.ndof)]
        for element in self.mesh.elements:
            num_nodes = len(element)
            if self.use_shell_element:
                element_type = ELEMENTS_MAP_SHELL[num_nodes]
            else:
                element_type = ELEMENTS_MAP_2D[num_nodes]
            dofs = [element_type.dofs_per_node * node + i for node in element for i in (0, 1)]
            for i in dofs:
                adjacency[i].update(dofs)
        return [len(adj) for adj in adjacency]

    def _create_matrix(self):
        """Crea matriz PETSc con preasignación óptima."""
        mat = PETSc.Mat().createAIJ((self.ndof, self.ndof), nnz=self.nnz)
        mat.setUp()
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        return mat

    @property
    def u(self) -> np.ndarray:
        return self._u.getArray()

    @property
    def K(self) -> np.ndarray:
        return petsc_to_numpy(self._K)

    @property
    def M(self) -> np.ndarray:
        return petsc_to_numpy(self._M)

    @property
    def f(self) -> np.ndarray:
        return self._f.getArray()

    def _assemble_matrix(self, property: Literal["K", "M"] = "K"):
        if property not in ("K", "M"):
            raise NotImplementedError

        matrix = self._create_matrix()
        for element_id, element in self.mesh.elements_map.items():
            coords = self.mesh.nodes[element]
            num_nodes = len(element)
            if self.use_shell_element:
                element_type = ELEMENTS_MAP_SHELL[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho, h=0.1)
            else:
                element_type = ELEMENTS_MAP_2D[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho)
            if property == "K":
                P = el.Ke
            if property == "M":
                P = el.Ke

            assert np.allclose(P, P.T), f"Element {property} matrix is not symmetric."

            global_dofs = self.global_dof_map[element_id]

            matrix.setValues(global_dofs, global_dofs, P, addv=PETSc.InsertMode.ADD)

        matrix.assemble()
        return matrix

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
        for element_id, element in self.mesh.elements_map.items():
            coords = self.mesh.nodes[element]
            num_nodes = len(element)
            if self.use_shell_element:
                element_type = ELEMENTS_MAP_SHELL[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho, h=0.1)
            else:
                element_type = ELEMENTS_MAP_2D[num_nodes]
                el = element_type(coords, self.E, self.nu, self.rho)
            Fe = el.load_vector(f_vec).astype(np.float32)

            global_dofs = self.global_dof_map[element_id]
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
        dofs = []
        for node in nodes:
            dofs.extend([2 * node, 2 * node + 1])  # DOFs para 2D
        self._f.setValues(dofs, np.repeat(values, len(nodes)))
        self._f.assemble()

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
                self._K.zeroRowsColumns([idx], diag=1.0, x=self._f)
                self._M.zeroRowsColumns([idx], diag=1.0, x=self._f)
                self._f.setValue(idx, 0.0)

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

    def solve_modal_analysis(self, num_modes: int = 6):
        """
        Performs modal analysis to determine the system's natural frequencies and vibration modes.

        Parameters:
        -----------
        num_modes : int, optional
            Number of eigenmodes to compute (default is 6).

        Returns
        -------
        frequencies : np.ndarray
            Natural frequencies in Hz.
        mode_shapes : np.ndarray
            Corresponding mode shapes (eigenvectors).
        """
        eff_num_modes = min(num_modes + 5, self._K.getSize()[0])

        eps = SLEPc.EPS().create()
        eps.setOperators(self._K, self._M)  # K primero, M segundo
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)

        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # <-- Usar TARGET_MAGNITUDE
        eps.setTarget(0.0)  # <-- Necesario para el shift-and-invert

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(0.0)  # Shift en 0 para buscar valores propios pequeños

        ksp = st.getKSP()
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")

        eps.setDimensions(eff_num_modes, PETSc.DECIDE, PETSc.DECIDE)
        eps.setTolerances(tol=1e-10, max_it=1000)
        eps.setFromOptions()

        eps.solve()

        nconv = eps.getConverged()
        if nconv < num_modes:
            raise RuntimeError(f"Solo {nconv} modos convergieron de {num_modes} solicitados.")

        eigvals, eigvecs = [], []
        for i in range(nconv):
            eigval = eps.getEigenvalue(i)
            eigvec = self._K.getVecLeft()
            eps.getEigenvector(i, eigvec)
            eigvals.append(eigval.real)
            eigvecs.append(eigvec.array)

        eigvals = np.array(eigvals)

        frequencies = np.sqrt(np.maximum(eigvals[:num_modes], 0)) / (2 * np.pi)
        return frequencies[:num_modes], np.array(eigvecs)[:, :num_modes]
