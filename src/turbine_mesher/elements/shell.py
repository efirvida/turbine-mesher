import itertools
from typing import List, Tuple

import numpy as np
from scipy.linalg import block_diag

# Constantes para índices de grados de libertad (DOFs)
U, V, W, THETA_X, THETA_Y, THETA_Z = 0, 1, 2, 3, 4, 5
DOFS_PER_NODE = 6  # u, v, w, theta_x, theta_y, theta_z


class MITC4:
    """Elemento shell MITC4 con estabilización de rigidez torsional (θ_z)."""

    def __init__(self, coords: np.ndarray, E: float, nu: float, t: float, alpha=1e-3):
        self.coords = coords
        self.E = E
        self.nu = nu
        self.t = t
        self.alpha = alpha  # Factor de estabilización para θ_z
        self.n_nodes = 4
        self.dofs = self.n_nodes * DOFS_PER_NODE
        self._K = np.zeros((self.dofs, self.dofs))
        self._gauss_order = 2  # Integración 2x2
        self._setup_tying_points()

    @property
    def integration_points(self) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Puntos y pesos de integración Gauss-Legendre."""
        x_1d, w_1d = np.polynomial.legendre.leggauss(self._gauss_order)
        points = list(itertools.product(x_1d, repeat=2))
        weights = [w1 * w2 for w1, w2 in itertools.product(w_1d, repeat=2)]
        return points, weights

    def _setup_tying_points(self) -> None:
        """Configura puntos de amarre MITC4."""
        self.tying_points = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
        self._precompute_shear_relations()

    def _compute_jacobian(
        self, dN_dxi: np.ndarray, dN_deta: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Calcula Jacobiano, su determinante e inversa."""
        J = np.array(
            [
                [dN_dxi @ self.coords[:, 0], dN_deta @ self.coords[:, 0]],
                [dN_dxi @ self.coords[:, 1], dN_deta @ self.coords[:, 1]],
            ]
        )
        det_J = np.linalg.det(J)
        if np.abs(det_J) < 1e-10:
            raise ValueError("Jacobiano singular.")
        inv_J = np.linalg.inv(J)
        return J, det_J, inv_J

    def _compute_cartesian_derivatives(
        self, dN_dxi: np.ndarray, dN_deta: np.ndarray, inv_J: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Derivadas cartesianas dN/dx y dN/dy."""
        dN_dx = inv_J[0, 0] * dN_dxi + inv_J[0, 1] * dN_deta
        dN_dy = inv_J[1, 0] * dN_dxi + inv_J[1, 1] * dN_deta
        return dN_dx, dN_dy

    def _precompute_shear_relations(self) -> None:
        """Precalcula matrices B de corte en puntos de amarre."""
        self.B_shear = []
        for xi, eta in self.tying_points:
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            _, _, inv_J = self._compute_jacobian(dN_dxi, dN_deta)
            dN_dx, dN_dy = self._compute_cartesian_derivatives(dN_dxi, dN_deta, inv_J)
            N = self.shape_functions(xi, eta)
            B = np.zeros((2, self.dofs))
            for i in range(4):
                B[0, DOFS_PER_NODE * i + W] = dN_dx[i]  # γ_xz = dw/dx - θ_y
                B[0, DOFS_PER_NODE * i + THETA_Y] = -N[i]
                B[1, DOFS_PER_NODE * i + W] = dN_dy[i]  # γ_yz = dw/dy + θ_x
                B[1, DOFS_PER_NODE * i + THETA_X] = N[i]
            self.B_shear.append(B)

    @property
    def C_membrane(self) -> np.ndarray:
        """Matriz constitutiva de membrana (tensión plana)."""
        factor = self.E * self.t / (1 - self.nu**2)
        return factor * np.array([[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]])

    @property
    def C_bending(self) -> np.ndarray:
        """Matriz constitutiva de flexión."""
        factor = self.E * self.t**3 / (12 * (1 - self.nu**2))
        return factor * np.array([[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]])

    @property
    def C_shear(self) -> np.ndarray:
        """Matriz constitutiva de cortante."""
        factor = self.E * self.t / (2 * (1 + self.nu))
        return factor * np.eye(2)

    def _add_drilling_stiffness(self) -> None:
        """Añade rigidez de estabilización al DOF θ_z (K_{66} para cada nodo)."""
        if self.alpha <= 0:
            return

        G = self.E / (2 * (1 + self.nu))
        A = self.area
        # Debug: Imprimir parámetros clave
        print(f"[DEBUG] alpha={self.alpha}, G={G}, A={A}, t={self.t}")
        k_drill = self.alpha * G * self.t * A
        print(f"[DEBUG] k_drill={k_drill}")

        dofs = DOFS_PER_NODE * np.arange(self.n_nodes) + THETA_Z
        print(f"[DEBUG] DOFs θ_z: {dofs}")
        self._K[dofs, dofs] += k_drill

    @property
    def area(self) -> float:
        points, weights = self.integration_points
        area = 0.0
        for (xi, eta), w in zip(points, weights):
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            J, det_J, _ = self._compute_jacobian(dN_dxi, dN_deta)
            area += det_J * w
        return area

    @property
    def Ke(self) -> np.ndarray:
        """Ensambla la matriz de rigidez del elemento."""
        self._K = np.zeros((self.dofs, self.dofs))  # Resetear matriz
        points, weights = self.integration_points

        # Ensamblar rigidez estándar
        for (xi, eta), w in zip(points, weights):
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            J, det_J, inv_J = self._compute_jacobian(dN_dxi, dN_deta)
            dN_dx, dN_dy = self._compute_cartesian_derivatives(dN_dxi, dN_deta, inv_J)

            B_membrane = self._build_membrane_B_matrix(dN_dx, dN_dy)
            B_bending = self._build_bending_B_matrix(dN_dx, dN_dy)
            B_shear = self._interpolate_shear(xi, eta)

            B = np.vstack([B_membrane, B_bending, B_shear])
            C = block_diag(self.C_membrane, self.C_bending, self.C_shear)
            self._K += (B.T @ C @ B) * det_J * w

        # Añadir rigidez de estabilización para θ_z
        self._add_drilling_stiffness()

        return self._K

    def _build_membrane_B_matrix(self, dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
        """Matriz B para deformación de membrana."""
        B = np.zeros((3, self.dofs))
        for i in range(4):
            B[0, DOFS_PER_NODE * i + U] = dN_dx[i]
            B[1, DOFS_PER_NODE * i + V] = dN_dy[i]
            B[2, DOFS_PER_NODE * i + U] = dN_dy[i]
            B[2, DOFS_PER_NODE * i + V] = dN_dx[i]
        return B

    def _build_bending_B_matrix(self, dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
        """Matriz B para curvatura de flexión."""
        B = np.zeros((3, self.dofs))
        for i in range(4):
            B[0, DOFS_PER_NODE * i + THETA_X] = dN_dx[i]
            B[1, DOFS_PER_NODE * i + THETA_Y] = dN_dy[i]
            B[2, DOFS_PER_NODE * i + THETA_X] = dN_dy[i]
            B[2, DOFS_PER_NODE * i + THETA_Y] = dN_dx[i]
        return B

    def _interpolate_shear(self, xi: float, eta: float) -> np.ndarray:
        """Interpola B_shear usando funciones MITC4."""
        N_mitc = [0.5 * (1 + eta), 0.5 * (1 - xi), 0.5 * (1 - eta), 0.5 * (1 + xi)]
        B_shear = np.zeros((2, self.dofs))
        for i in range(4):
            B_shear += N_mitc[i] * self.B_shear[i]
        return B_shear

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Funciones de forma bilineales."""
        return 0.25 * np.array(
            [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
        )

    def shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Derivadas de las funciones de forma."""
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return dN_dxi, dN_deta


if __name__ == "__main__":
    # Ejemplo de uso
    from turbine_mesher.helpers import array2str

    coords = np.array([[0, 0], [3, 0], [2, 1], [0, 1]])
    element = MITC4(coords=coords, E=1, nu=0.5, t=1, alpha=1e-3)
    K = element.Ke
    print("Matriz de Rigidez K (primer nodo):\n\n", array2str("K", K[:6, :6]))
