from typing import Tuple

import numpy as np

from turbine_mesher.elements.abc import Element2D

__all__ = [
    "TriElement",
    "TriQuadElement",
]


class TriElement(Element2D):
    """
    Linear triangular element with 3 nodes for 2D finite element analysis.
    """

    def __init__(self, coords: np.ndarray, E: float, nu: float, density: float = 1):
        """
        Initialize the linear triangular element with nodal coordinates.
        """
        super().__init__(coords, E, nu, density)  # 1 Gauss point

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta
        return np.array([N1, N2, N3])

    def shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        dN_dxi = np.array([-1, 1, 0])
        dN_deta = np.array([-1, 0, 1])
        return dN_dxi, dN_deta

    @property
    def integration_points(self):
        if self._integration_points == 1:
            # Regla de 1 punto (exacta para polinomios de orden ≤ 1)
            return ([(1 / 3, 1 / 3)], [0.5])

        elif self._integration_points == 3:
            # Regla de 3 puntos (exacta para polinomios de orden ≤ 2)
            return (
                [
                    (1 / 6, 1 / 6),
                    (2 / 3, 1 / 6),
                    (1 / 6, 2 / 3),
                ],
                [1 / 6, 1 / 6, 1 / 6],
            )

        elif self._integration_points == 4:
            # Regla de 4 puntos (exacta para polinomios de orden ≤ 3)
            return (
                [
                    (1 / 3, 1 / 3),
                    (1 / 5, 1 / 5),
                    (3 / 5, 1 / 5),
                    (1 / 5, 3 / 5),
                ],
                [-27 / 96, 25 / 96, 25 / 96, 25 / 96],
            )  # Dunavant de 4 puntos

        elif self._integration_points == 7:
            # Regla de 7 puntos (exacta para polinomios de orden ≤ 5)
            return (
                [
                    (1 / 3, 1 / 3),
                    (0.0597158717, 0.4701420641),
                    (0.4701420641, 0.0597158717),
                    (0.4701420641, 0.4701420641),
                    (0.7974269853, 0.1012865073),
                    (0.1012865073, 0.7974269853),
                    (0.1012865073, 0.1012865073),
                ],
                [
                    0.2250000000,
                    0.1323941527,
                    0.1323941527,
                    0.1323941527,
                    0.1259391805,
                    0.1259391805,
                    0.1259391805,
                ],
            )  # Dunavant de 7 puntos

        else:
            raise NotImplementedError(
                f"Integration rule for {self._integration_points} points is not implemented."
            )

    @property
    def area(self) -> float:
        x1, y1 = self.coords[0]
        x2, y2 = self.coords[1]
        x3, y3 = self.coords[2]
        return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


class TriQuadElement(TriElement):
    """
    Quadratic triangular element with 6 nodes for 2D finite element analysis.
    """

    def __init__(self, coords: np.ndarray, E: float, nu: float, density: float = 1):
        """
        Initialize the linear triangular element with nodal coordinates.
        """
        super().__init__(coords, E, nu, density)
        self._integration_points = 7

    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        N1 = (1 - xi - eta) * (1 - 2 * xi - 2 * eta)
        N2 = xi * (2 * xi - 1)
        N3 = eta * (2 * eta - 1)
        N4 = 4 * xi * (1 - xi - eta)
        N5 = 4 * xi * eta
        N6 = 4 * eta * (1 - xi - eta)

        return np.array([N1, N2, N3, N4, N5, N6])

    def shape_function_derivatives(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        dN_dxi = np.array(
            [
                -3 + 4 * xi + 4 * eta,
                4 * xi - 1,
                0,
                4 - 8 * xi - 4 * eta,
                4 * eta,
                -4 * eta,
            ]
        )

        dN_deta = np.array(
            [
                -3 + 4 * xi + 4 * eta,
                0,
                4 * eta - 1,
                -4 * xi,
                4 * xi,
                4 - 4 * xi - 8 * eta,
            ]
        )

        return dN_dxi, dN_deta
