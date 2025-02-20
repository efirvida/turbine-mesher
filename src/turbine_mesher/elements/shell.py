import numpy as np
from scipy.linalg import block_diag

from turbine_mesher.elements.quad import QuadElement

# Degree of Freedom (DOF) indices for node displacement components
U, V, W, THETA_X, THETA_Y, THETA_Z = 0, 1, 2, 3, 4, 5


class MITC4(QuadElement):
    """
    MITC4 shell element with drilling DOF stabilization (θ_z).

    Implements the Mixed Interpolation of Tensorial Components (MITC) formulation
    for quadrilateral shell elements to prevent shear locking. Includes stabilization
    for the drilling rotation DOF using a small stiffness factor (alpha).

    Attributes:
        h (float): Shell thickness
        alpha (float): Stabilization factor for drilling rotation (θ_z)
        _gauss_order (int): Integration order for full quadrature
        _tying_points (list): MITC4 shear strain tying points in parametric coordinates
        _B_shear (list): Precomputed shear strain-displacement matrices at tying points

    Reference:
        Bathe, K.J., and Dvorkin, E.N. (1985). "A four-node plate bending element
        based on Mindlin/Reissner plate theory and a mixed interpolation."
    """

    dofs_per_node = 6  # u, v, w, θ_x, θ_y, θ_z

    def __init__(self, coords: np.ndarray, E: float, nu: float, rho: float, h: float, alpha=1e-3):
        """
        Initialize MITC4 shell element.

        Args:
            coords: Nodal coordinates matrix (4 nodes x 2 coordinates)
            E: Young's modulus
            nu: Poisson's ratio
            rho: Material density
            h: Shell thickness
            alpha: Stabilization factor for drilling DOF (default: 1e-3)
        """
        super().__init__(coords, E, nu, rho)
        self.h = h
        self.alpha = alpha
        self._gauss_order = 2  # Full integration order
        self._setup_tying_points()

    def _setup_tying_points(self) -> None:
        """Initialize MITC4 shear strain tying points in parametric coordinates."""
        self._tying_points = [
            (1.0, 0.0),  # Midpoint of edge ξ=1
            (0.0, 1.0),  # Midpoint of edge η=1
            (-1.0, 0.0),  # Midpoint of edge ξ=-1
            (0.0, -1.0),  # Midpoint of edge η=-1
        ]
        self._precompute_shear_relations()

    @property
    def C_membrane(self) -> np.ndarray:
        """
        Membrane constitutive matrix (plane stress).

        Returns:
            (3x3 array): Material matrix calculated as:
            C_mem = (E*h)/(1-ν²) * [[1,  ν,     0    ],
                                    [ν,  1,     0    ],
                                    [0,  0, (1-ν)/2 ]]
        """
        factor = self.E * self.h / (1 - self.nu**2)
        return factor * np.array([[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]])

    @property
    def C_bending(self) -> np.ndarray:
        """
        Bending constitutive matrix.

        Returns:
            (3x3 array): Material matrix calculated as:
            C_bend = (E*h³)/(12(1-ν²)) * [[1,  ν,     0    ],
                                          [ν,  1,     0    ],
                                          [0,  0, (1-ν)/2 ]]
        """
        factor = self.E * self.h**3 / (12 * (1 - self.nu**2))
        return factor * np.array([[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]])

    @property
    def C_shear(self) -> np.ndarray:
        """
        Shear constitutive matrix.

        Returns:
            (2x2 array): Material matrix calculated as:
            C_shear = (E*h)/(2(1+ν)) * [[1, 0],
                                        [0, 1]]
        """
        k = 5 / 6  # shear-correction factor
        factor = self.E * self.h * k / (2 * (1 + self.nu))
        return factor * np.eye(2)

    def _add_drilling_stiffness(self) -> None:
        """
        Add stabilization stiffness to drilling rotation DOFs (θ_z).

        The stabilization stiffness is calculated as:
            k_drill = α * G * h * A
        where:
            α = stabilization factor (alpha),
            G = shear modulus (E/(2(1+ν))),
            A = element area,
            h = shell thickness

        The stiffness is added to the diagonal entries corresponding to θ_z DOFs.
        """
        if self.alpha <= 0:
            return

        G = self.E / (2 * (1 + self.nu))  # Shear modulus
        A = self.area  # Element area
        k_drill = self.alpha * G * self.h * A

        # Add to θ_z DOFs (positions 5, 11, 17, 23 in 24x24 matrix)
        dofs = self.dofs_per_node * np.arange(self.n_nodes) + THETA_Z
        self._K[dofs, dofs] += k_drill

    @property
    def area(self) -> float:
        """Calculate element area through numerical integration."""
        area = 0.0
        points, weights = self.integration_points
        for (xi, eta), w in zip(points, weights):
            _, det_J, _ = self._compute_jacobian(xi, eta)
            area += det_J * w
        return area

    @property
    def Ke(self) -> np.ndarray:
        """
        Assemble the element stiffness matrix using MITC4 formulation.

        The stiffness matrix integrates contributions from:
        - Membrane strains
        - Bending strains
        - Shear strains (with MITC interpolation)
        - Drilling DOF stabilization

        Returns:
            (24x24 array): Element stiffness matrix
        """
        self._K = np.zeros((self.dofs, self.dofs))  # Reset stiffness matrix
        points, weights = self.integration_points

        # Numerical integration loop
        for (xi, eta), w in zip(points, weights):
            # Compute shape function derivatives and Jacobian
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            _, det_J, inv_J = self._compute_jacobian(xi, eta)
            dN_dx, dN_dy = self._compute_cartesian_derivatives(dN_dxi, dN_deta, inv_J)

            # Strain-displacement matrices
            B_membrane = self._build_membrane_B_matrix(dN_dx, dN_dy)
            B_bending = self._build_bending_B_matrix(dN_dx, dN_dy)
            B_shear = self._interpolate_shear_strains(xi, eta)

            # Combine strain components and integrate
            B = np.vstack([B_membrane, B_bending, B_shear])
            C = block_diag(self.C_membrane, self.C_bending, self.C_shear)
            self._K += (B.T @ C @ B) * det_J * w

        self._add_drilling_stiffness()  # Add drilling DOF stabilization
        return self._K

    def _build_membrane_B_matrix(self, dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
        """
        Construct membrane strain-displacement matrix.

        Args:
            dN_dx: Shape function x-derivatives (4 elements)
            dN_dy: Shape function y-derivatives (4 elements)

        Returns:
            (3x24 array): B matrix relating membrane strains to nodal displacements
                          ε = [ε_x, ε_y, γ_xy]^T = B_mem * u
        """
        B = np.zeros((3, self.dofs))
        for i in range(4):
            # Membrane strain contributions from u and v displacements
            B[0, self.dofs_per_node * i + U] = dN_dx[i]  # ε_x = du/dx
            B[1, self.dofs_per_node * i + V] = dN_dy[i]  # ε_y = dv/dy
            B[2, self.dofs_per_node * i + U] = dN_dy[i]  # γ_xy = du/dy + dv/dx
            B[2, self.dofs_per_node * i + V] = dN_dx[i]
        return B

    def load_vector(self, body_force: np.ndarray) -> np.ndarray:
        """
        Assemble the load vector for the element due to a constant body force.

        The load vector is computed as:

            .. math::
                f_e = \\int_{\\Omega} N^T \\mathbf{b} \\; d\\Omega,

        where:
            - \\(N\\) is the shape function matrix for MITC4 elements,
            - \\(\\mathbf{b}\\) is the constant body force vector (shape (3,)).

        Parameters
        ----------
        body_force : np.ndarray
            A 3-element array representing the body force per unit volume (in 3D).

        Returns
        -------
        np.ndarray
            Element load vector of shape (dofs,).
        """
        points, weights = self.integration_points
        f = np.zeros(self.dofs)
        for (xi, eta), w in zip(points, weights):
            _, det_J, _ = self._compute_jacobian(xi, eta)
            N_values = self.shape_functions(xi, eta)
            N_matrix = np.zeros((3, self.dofs))

            # Construir N_matrix considerando 3 traslaciones por nodo
            N_matrix[0, 0::6] = N_values  # Desplazamiento en X
            N_matrix[1, 1::6] = N_values  # Desplazamiento en Y
            N_matrix[2, 2::6] = N_values  # Desplazamiento en Z

            # Nota: Para MITC4, generalmente no hay contribuciones directas a las rotaciones
            # por fuerzas de cuerpo constantes, pero si fuese necesario, se incluirían así:
            # N_matrix[0, 3::6] = N_values  # Rotación alrededor de X
            # N_matrix[1, 4::6] = N_values  # Rotación alrededor de Y
            # N_matrix[2, 5::6] = N_values  # Rotación alrededor de Z

            f += (N_matrix.T @ body_force) * det_J * w
        return f

    def _build_bending_B_matrix(self, dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
        """
        Construct bending strain-displacement matrix.

        Args:
            dN_dx: Shape function x-derivatives (4 elements)
            dN_dy: Shape function y-derivatives (4 elements)

        Returns:
            (3x24 array): B matrix relating bending strains to nodal rotations
                          κ = [κ_x, κ_y, κ_xy]^T = B_bend * θ
        """
        B = np.zeros((3, self.dofs))
        for i in range(4):
            # Bending strain contributions from θ_x and θ_y rotations
            B[0, self.dofs_per_node * i + THETA_X] = dN_dx[i]  # κ_x = dθ_x/dx
            B[1, self.dofs_per_node * i + THETA_Y] = dN_dy[i]  # κ_y = dθ_y/dy
            B[2, self.dofs_per_node * i + THETA_X] = dN_dy[i]  # κ_xy = dθ_x/dy + dθ_y/dx
            B[2, self.dofs_per_node * i + THETA_Y] = dN_dx[i]
        return B

    def _precompute_shear_relations(self) -> None:
        """Precompute shear strain-displacement matrices at tying points."""
        self._B_shear = []
        for xi, eta in self._tying_points:
            # Compute shape function derivatives at tying point
            dN_dxi, dN_deta = self.shape_function_derivatives(xi, eta)
            _, _, inv_J = self._compute_jacobian(xi, eta)
            dN_dx, dN_dy = self._compute_cartesian_derivatives(dN_dxi, dN_deta, inv_J)
            N = self.shape_functions(xi, eta)

            # Build shear B matrix for current tying point
            B_shear = np.zeros((2, self.dofs))
            for i in range(4):
                # γ_xz = dw/dx - θ_y
                B_shear[0, self.dofs_per_node * i + W] = dN_dx[i]
                B_shear[0, self.dofs_per_node * i + THETA_Y] = -N[i]

                # γ_yz = dw/dy + θ_x
                B_shear[1, self.dofs_per_node * i + W] = dN_dy[i]
                B_shear[1, self.dofs_per_node * i + THETA_X] = N[i]

            self._B_shear.append(B_shear)

    def _interpolate_shear_strains(self, xi: float, eta: float) -> np.ndarray:
        """
        Interpolate shear strains using MITC4 tying scheme.

        Args:
            xi: Parametric ξ coordinate
            eta: Parametric η coordinate

        Returns:
            (2x24 array): Interpolated shear strain-displacement matrix
        """
        # MITC4 interpolation functions
        N_mitc = [
            0.5 * (1 + eta),  # For tying point at ξ=1
            0.5 * (1 - xi),  # For tying point at η=1
            0.5 * (1 - eta),  # For tying point at ξ=-1
            0.5 * (1 + xi),  # For tying point at η=-1
        ]

        # Linear combination of precomputed B matrices
        B_shear = np.zeros((2, self.dofs))
        for i in range(4):
            B_shear += N_mitc[i] * self._B_shear[i]
        return B_shear

    @property
    def Me(self) -> np.ndarray:
        """
        Assemble consistent mass matrix considering translational and rotational inertia.

        The mass matrix is computed as:
            M = ∫[ρ*h*(N_u^T N_u + N_v^T N_v + N_w^T N_w) +
                 ρ*h³/12*(N_θx^T N_θx + N_θy^T N_θy)] dA

        where:
            N_u, N_v, N_w = Translational shape functions
            N_θx, N_θy = Rotational shape functions
            ρ = Material density
            h = Shell thickness

        Returns:
            (24x24 array): Consistent mass matrix
        """
        M = np.zeros((self.dofs, self.dofs))
        points, weights = self.integration_points

        for (xi, eta), w in zip(points, weights):
            # Jacobian and shape functions at integration point
            _, det_J, _ = self._compute_jacobian(xi, eta)
            N = self.shape_functions(xi, eta)

            # Translational inertia components (u, v, w)
            M_trans = np.zeros((self.dofs, self.dofs))
            for i, dof in enumerate([U, V, W]):
                N_dof = np.zeros(self.dofs)
                N_dof[dof :: self.dofs_per_node] = N
                M_trans += np.outer(N_dof, N_dof)

            # Rotational inertia components (θ_x, θ_y)
            M_rot = np.zeros((self.dofs, self.dofs))
            for i, dof in enumerate([THETA_X, THETA_Y]):
                N_dof = np.zeros(self.dofs)
                N_dof[dof :: self.dofs_per_node] = N
                M_rot += np.outer(N_dof, N_dof)

            # Accumulate contributions
            M += self.rho * self.h * M_trans * det_J * w
            M += self.rho * (self.h**3 / 12) * M_rot * det_J * w

        self._M = M
        return self._M


if __name__ == "__main__":
    # Example usage
    from turbine_mesher.helpers import array2str

    coords = np.array([[0, 0], [3, 0], [2, 1], [0, 1]])
    element = MITC4(coords=coords, E=1, nu=0.5, rho=3000, h=1, alpha=1e-3)
    K = element.Ke
    M = element.Me
    print(f"Element type: {element.element_type}")
    print("Stiffness matrix (first node block):\n", array2str("K", K[:6, :6]))
    print("Mass matrix (first node block):\n", array2str("M", M[:6, :6]))
