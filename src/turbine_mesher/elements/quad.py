import numpy as np

from turbine_mesher.elements.abc import Element2D
from turbine_mesher.helpers import quad_area

__all__ = ["QuadElement", "SerendipityElement", "LagrangeElement"]


class QuadElement(Element2D):
    """
    4-node isoparametric quadrilateral element for finite element analysis.

    Attributes
    ----------
    coords : np.ndarray
        Node coordinates matrix of shape (4, 3) with ordering:
        Node 0 (bottom-left), Node 1 (bottom-right), Node 2 (top-right), Node 3 (top-left)
    """

    def __init__(self, coords: np.ndarray, E: float, nu: float, rho: float = 1):
        """
        Initialize quadrilateral element.

        Parameters
        ----------
        coords : np.ndarray
            Node coordinates in counter-clockwise order starting from bottom-left:

                3 -------- 2
                |          |
                |          |
                0 -------- 1

            Must be shape (4, 3) or (4, 2)
        """

        super().__init__(coords, E, nu, rho)
        self._gauss_order = 2

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute bilinear shape functions for quadrilateral element.

        Parameters
        ----------
        xi : float
            Natural coordinate in [-1, 1]
        eta : float
            Natural coordinate in [-1, 1]

        Returns
        -------
        np.ndarray
            Array of shape functions [N0, N1, N2, N3]
        """
        return 0.25 * np.array(
            [
                (1 - xi) * (1 - eta),  # N0
                (1 + xi) * (1 - eta),  # N1
                (1 + xi) * (1 + eta),  # N2
                (1 - xi) * (1 + eta),  # N3
            ]
        )

    def shape_function_derivatives(self, xi: float, eta: float):
        """
        Compute derivatives of bilinear shape functions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Derivatives w.r.t. xi and eta as (dN_dxi, dN_deta)
        """
        dN_dxi = [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]
        dN_deta = [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]

        return 0.25 * np.array([dN_dxi, dN_deta])

    @property
    def area(self) -> float:
        """
        Compute the area of a quadrilateral element.

        Supports 4-node (bilinear), 8-node (serendipity), and 9-node (Lagrange) elements.
        The 4-node element uses the shoelace formula, while the 8-node and 9-node elements
        are approximated by dividing them into 4 triangles.

        Returns
        -------
        float
            The approximate area of the quadrilateral element.

        Raises
        ------
        ValueError
            If `coords` does not have a valid shape.
        """
        if self.coords.shape not in [(4, 3), (8, 3)]:
            raise ValueError("Input array must have shape (4,3), (8,3).")
        return quad_area(self.coords)


class SerendipityElement(QuadElement):
    """
    8-node serendipity element for finite element analysis.

    Attributes
    ----------
    coords : np.ndarray
        Node coordinates matrix of shape (8, 3) with ordering:
        Node 0 (bottom-left), Node 1 (bottom-right), Node 2 (top-right),
        Node 3 (top-left), Node 4 (bottom-mid), Node 5 (right-mid),
        Node 6 (top-mid), Node 7 (left-mid)
    """

    def __init__(self, coords: np.ndarray, E: float, nu: float, rho: float = 1):
        """
        Initialize the 8-node serendipity element.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of the 8 nodes, ordered as:

                3 --- 6 --- 2
                |           |
                7           5
                |           |
                0 --- 4 --- 1

             Must be shape (8, 3) or (8,2)

        """
        super().__init__(coords, E, nu, rho)

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Evaluate the 8 serendipity shape functions at (xi, eta).

        Returns
        -------
        np.ndarray
            Array of shape functions [N0, N1, ..., N7]
        """

        N0 = 0.25 * (1 - xi) * (1 - eta) * (-1 - xi - eta)
        N1 = 0.25 * (1 + xi) * (1 - eta) * (-1 + xi - eta)
        N2 = 0.25 * (1 + xi) * (1 + eta) * (-1 + xi + eta)
        N3 = 0.25 * (1 - xi) * (1 + eta) * (-1 - xi + eta)
        N4 = 0.5 * (1 - xi**2) * (1 - eta)
        N5 = 0.5 * (1 + xi) * (1 - eta**2)
        N6 = 0.5 * (1 - xi**2) * (1 + eta)
        N7 = 0.5 * (1 - xi) * (1 - eta**2)
        return np.array([N0, N1, N2, N3, N4, N5, N6, N7])

    def shape_function_derivatives(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute derivatives of the serendipity shape functions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Derivatives w.r.t. xi and eta as (dN_dxi, dN_deta)
        """

        dN_dxi = [
            0.25 * (1 - eta) * (2 * xi + eta),
            0.25 * (1 - eta) * (2 * xi - eta),
            0.25 * (1 + eta) * (2 * xi + eta),
            0.25 * (1 + eta) * (2 * xi - eta),
            -xi * (1 - eta),
            0.5 * (1 - eta**2),
            -xi * (1 + eta),
            -0.5 * (1 - eta**2),
        ]

        dN_deta = [
            0.25 * (1 - xi) * (xi + 2 * eta),
            0.25 * (1 + xi) * (-xi + 2 * eta),
            0.25 * (1 + xi) * (xi + 2 * eta),
            0.25 * (1 - xi) * (-xi + 2 * eta),
            -0.5 * (1 - xi**2),
            -(1 + xi) * eta,
            0.5 * (1 - xi**2),
            -(1 - xi) * eta,
        ]

        return np.array([dN_dxi, dN_deta])


class LagrangeElement(QuadElement):
    """
    9-node Lagrange element for finite element analysis.

    Attributes
    ----------
    coords : np.ndarray
        Node coordinates matrix of shape (9, 3) with ordering:
        Node 0 (bottom-left), Node 1 (bottom-right), Node 2 (top-right),
        Node 3 (top-left), Node 4 (bottom-mid), Node 5 (right-mid),
        Node 6 (top-mid), Node 7 (left-mid), Node 8 (center)
    """

    def __init__(self, coords: np.ndarray, E: float, nu: float, rho: float = 1):
        """
        Initialize the 9-node Lagrange element.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of the 9 nodes, ordered as:

                3 --- 6 --- 2
                |           |
                7     8     5
                |           |
                0 --- 4 --- 1

            Must be shape (9, 2) or (9, 3)

        """
        super().__init__(coords, E, nu, rho)
        self._gauss_order = 3

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """
        Evaluate the 9 Lagrange shape functions at (xi, eta).

        Returns
        -------
        np.ndarray
            Array of shape functions [N0, N1, ..., N8]
        """
        N0 = 0.25 * xi * (xi - 1) * eta * (eta - 1)
        N1 = 0.25 * xi * (xi + 1) * eta * (eta - 1)
        N2 = 0.25 * xi * (xi + 1) * eta * (eta + 1)
        N3 = 0.25 * xi * (xi - 1) * eta * (eta + 1)

        N4 = 0.5 * (1 - xi**2) * eta * (eta - 1)
        N5 = 0.5 * xi * (xi + 1) * (1 - eta**2)
        N6 = 0.5 * (1 - xi**2) * eta * (eta + 1)
        N7 = 0.5 * xi * (xi - 1) * (1 - eta**2)

        N8 = (1 - xi**2) * (1 - eta**2)

        return np.array([N0, N1, N2, N3, N4, N5, N6, N7, N8])

    def shape_function_derivatives(self, xi: float, eta: float) -> np.ndarray:
        """
        Compute derivatives of the 9-node Lagrange shape functions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Derivatives w.r.t. xi and eta as (dN_dxi, dN_deta)
        """
        dN_dxi = np.array(
            [
                0.25 * (2 * xi - 1) * eta * (eta - 1),  # N0
                0.25 * (2 * xi + 1) * eta * (eta - 1),  # N1
                0.25 * (2 * xi + 1) * eta * (eta + 1),  # N2
                0.25 * (2 * xi - 1) * eta * (eta + 1),  # N3
                -xi * eta * (eta - 1),  # N4
                0.5 * (2 * xi + 1) * (1 - eta**2),  # N5
                -xi * eta * (eta + 1),  # N6
                0.5 * (2 * xi - 1) * (1 - eta**2),  # N7
                -2 * xi * (1 - eta**2),  # N8
            ]
        )

        dN_deta = np.array(
            [
                0.25 * xi * (xi - 1) * (2 * eta - 1),  # N0
                0.25 * xi * (xi + 1) * (2 * eta - 1),  # N1
                0.25 * xi * (xi + 1) * (2 * eta + 1),  # N2
                0.25 * xi * (xi - 1) * (2 * eta + 1),  # N3
                0.5 * (1 - xi**2) * (2 * eta - 1),  # N4
                -xi * (xi + 1) * eta,  # N5
                0.5 * (1 - xi**2) * (2 * eta + 1),  # N6
                -xi * (xi - 1) * eta,  # N7
                -2 * eta * (1 - xi**2),  # N8
            ]
        )

        return np.array([dN_dxi, dN_deta])
