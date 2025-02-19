from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from turbine_mesher.helpers import gauss_legendre_quadrature


class Element2D(ABC):
    """
    Abstract base class for 2D finite element types.

    This class provides formulations for the stiffness matrix, the consistent mass matrix,
    and the load vector (body force) using an isoparametric finite element approach.

    The stiffness matrix is computed as:

        .. math::
            K_e = \\int_{\\Omega} B^T C B \\; d\\Omega,

    the consistent mass matrix as:

        .. math::
            M_e = \\int_{\\Omega} \\rho N^T N \\; d\\Omega,

    and the load vector as:

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

    dofs_per_node = 2

    def __init__(self, coords: np.ndarray, E: float, nu: float, rho: float = 1.0):
        """
        Initialize the element with nodal coordinates and material properties.

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates array of shape (n_nodes, 2).
        E : float
            Young's modulus of the material.
        nu : float
            Poisson's ratio of the material.
        rho : float, optional
            Material density (default is 1.0).
        """
        self.coords = coords
        self.n_nodes = coords.shape[0]
        self.E = E
        self.nu = nu
        self.rho = rho
        self._K = np.zeros((self.dofs, self.dofs))
        self._M = np.zeros((self.dofs, self.dofs))
        self._f = np.zeros(self.dofs)
        self._gauss_order = 1  # Number of Gauss integration points

    @property
    def dofs(self) -> int:
        """
        Total number of degrees of freedom for the element.

        Returns
        -------
        int
            Total degrees of freedom (n_nodes * dofs_per_node).
        """
        return self.dofs_per_node * self.n_nodes

    @property
    def integration_points(self):
        """
        Get integration points and weights for numerical integration.

        Assumes that the function `gauss_legendre_quadrature` returns a tuple:
            (points, weights)

        Returns
        -------
        tuple
            A tuple containing:
                - points: an array of integration points,
                - weights: an array of corresponding weights.
        """
        return gauss_legendre_quadrature(self._gauss_order)

    @property
    def C(self) -> np.ndarray:
        """
        Constitutive (elasticity) matrix in plane stress conditions.

        Returns
        -------
        np.ndarray
            Elasticity matrix of shape (3, 3).

        Notes
        -----
        The constitutive matrix is given by:

            .. math::
                C = \\begin{bmatrix}
                    \\lambda + 2\\mu & \\lambda & 0 \\\\
                    \\lambda & \\lambda + 2\\mu & 0 \\\\
                    0 & 0 & \\mu
                \\end{bmatrix},

        where:
            - \\(\\lambda = \\frac{E \\nu}{(1+\\nu)(1-2\\nu)}\\),
            - \\(\\mu = \\frac{E}{2(1+\\nu)}\\).
        """
        lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        return np.array(
            [
                [lambda_ + 2 * mu, lambda_, 0],
                [lambda_, lambda_ + 2 * mu, 0],
                [0, 0, mu],
            ]
        )

    @property
    def Ke(self) -> np.ndarray:
        """
        Compute the element stiffness matrix.

        The stiffness matrix is computed as:

            .. math::
                K_e = \\int_{\\Omega} B^T C B \\; d\\Omega,

        where:
            - \\(B\\) is the strain-displacement matrix,
            - \\(C\\) is the constitutive matrix.

        Returns
        -------
        np.ndarray
            Element stiffness matrix of shape (dofs x dofs).
        """
        points, weights = self.integration_points

        for (xi, eta), w in zip(points, weights):
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            _, det_J, inv_J = self._compute_jacobian(xi, eta)
            dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta
            dN_dy = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta

            B = np.zeros((3, self.dofs))
            B[0, 0::2] = dN_dx  # ε_xx
            B[1, 1::2] = dN_dy  # ε_yy
            B[2, 0::2] = dN_dy  # γ_xy
            B[2, 1::2] = dN_dx  # γ_xy

            self._K += (B.T @ self.C @ B) * det_J * w

        return self._K

    @property
    def Me(self) -> np.ndarray:
        """
        Compute the consistent mass matrix.

        The mass matrix is computed as:

            .. math::
                M_e = \\int_{\\Omega} \\rho N^T N \\; d\\Omega,

        where:
            - \\(N\\) is the shape function matrix assembled for two degrees of freedom per node,
            - \\(\\rho\\) is the material density,
            - \\(\\Omega\\) is the element domain.

        Returns
        -------
        np.ndarray
            Element mass matrix of shape (dofs x dofs).
        """
        points, weights = self.integration_points
        for (xi, eta), w in zip(points, weights):
            _, det_J, _ = self._compute_jacobian(xi, eta)
            N_values = self.shape_functions(xi, eta)  # Array of shape (n_nodes,)

            N_matrix = np.zeros((2, self.dofs))
            N_matrix[0, 0::2] = N_values
            N_matrix[1, 1::2] = N_values

            self._M += self.rho * (N_matrix.T @ N_matrix) * det_J * w

        return self._M

    @property
    def element_type(self) -> str:
        """
        Get the type of the element.

        Returns
        -------
        str
            A string indicating the element type (e.g., 'Triangular', 'Quadrilateral').
        """
        return self.__class__.__name__

    def load_vector(self, body_force: np.ndarray) -> np.ndarray:
        """
        Assemble the load vector for the element due to a constant body force.

        The load vector is computed as:

            .. math::
                f_e = \\int_{\\Omega} N^T \\mathbf{b} \\; d\\Omega,

        where:
            - \\(N\\) is the shape function matrix assembled for two degrees of freedom per node,
            - \\(\\mathbf{b}\\) is the constant body force vector (shape (2,)).

        Parameters
        ----------
        body_force : np.ndarray
            A 2-element array representing the body force per unit volume (or area in 2D).

        Returns
        -------
        np.ndarray
            Element load vector of shape (dofs,).
        """
        points, weights = self.integration_points
        for (xi, eta), w in zip(points, weights):
            _, det_J, _ = self._compute_jacobian(xi, eta)
            N_values = self.shape_functions(xi, eta)
            N_matrix = np.zeros((2, self.dofs))
            N_matrix[0, 0::2] = N_values
            N_matrix[1, 1::2] = N_values
            self._f += (N_matrix.T @ body_force) * det_J * w
        return self._f

    def plot_element(self):
        """
        Visualize the finite element in 2D, displaying nodes and connectivity.

        This method plots the element by showing the node locations and connecting them with lines.
        """
        x = self.coords[:, 0]
        y = self.coords[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(x, y, c="r", label="Nodes")

        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi + 0.01, yi + 0.01, str(i), fontsize=12)

        num_nodes = len(self.coords)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                ax.plot([x[i], x[j]], [y[i], y[j]], "b-", lw=1)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.grid(True)
        plt.show()

    def _compute_jacobian(self, xi: float, eta: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute the Jacobian matrix, its determinant, and its inverse at given natural coordinates.

        Parameters
        ----------
        xi : float
            Natural coordinate xi.
        eta : float
            Natural coordinate eta.

        Returns
        -------
        tuple
            A tuple (J, det_J, inv_J) where:
                - J is the Jacobian matrix,
                - det_J is the determinant of J,
                - inv_J is the inverse of J.

        Raises
        ------
        ValueError
            If the determinant of the Jacobian is non-positive.
        """
        dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
        J = np.array(
            [
                [dN_dxi @ self.coords[:, 0], dN_deta @ self.coords[:, 0]],
                [dN_dxi @ self.coords[:, 1], dN_deta @ self.coords[:, 1]],
            ]
        )
        det_J = np.linalg.det(J)
        if det_J <= 0:
            raise ValueError("Jacobian determinant is non-positive. Check the node orientation.")
        inv_J = np.linalg.inv(J)
        return J, det_J, inv_J

    def _compute_cartesian_derivatives(
        self, dN_dxi: np.ndarray, dN_deta: np.ndarray, inv_J: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Derivadas cartesianas dN/dx y dN/dy."""
        dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta
        dN_dy = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta
        return dN_dx, dN_dy

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Compute the area of the element.

        Returns
        -------
        float
            Approximate area of the element.
        """
        pass

    @abstractmethod
    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute the shape functions at given natural coordinates.

        Parameters
        ----------
        xi : float
            Natural coordinate xi.
        eta : float
            Natural coordinate eta.

        Returns
        -------
        np.ndarray
            Array of shape functions evaluated at (xi, eta) with shape (n_nodes,).
        """
        pass

    @abstractmethod
    def shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the derivatives of the shape functions with respect to the natural coordinates.

        Parameters
        ----------
        xi : float
            Natural coordinate xi.
        eta : float
            Natural coordinate eta.

        Returns
        -------
        tuple
            A tuple containing:
                - dN_dxi: Derivative of shape functions with respect to xi (array of shape (n_nodes,)).
                - dN_deta: Derivative of shape functions with respect to eta (array of shape (n_nodes,)).
        """
        pass
