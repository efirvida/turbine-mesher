import itertools
from typing import Dict, List, Tuple

import meshio
import numpy as np

from .enums import ElementsTypes
from .types import PyNuMADMesh


def gauss_legendre_quadrature(n: int) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Compute Gauss-Legendre quadrature points and weights for 2D integration.

    Parameters
    ----------
    n : int
        Number of integration points in each dimension.

    Returns
    -------
    points : List[Tuple[float, float]]
        List of integration points in (xi, eta) coordinates.
    weights : List[float]
        Corresponding weights for each integration point.

    Notes
    -----
    Creates a tensor product grid of 1D Gauss-Legendre quadrature points.
    """
    x_1d, w_1d = np.polynomial.legendre.leggauss(n)
    points = list(itertools.product(x_1d, repeat=2))
    weights = [w1 * w2 for w1, w2 in itertools.product(w_1d, repeat=2)]
    return points, weights


def compute_triangular_element_normals(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, normalized: bool = True
) -> np.ndarray:
    """
    Calculate the normal vector of a triangular element in 3D space.

    Parameters
    ----------
    p1 : np.ndarray
        First vertex coordinates, shape (3,)
    p2 : np.ndarray
        Second vertex coordinates, shape (3,)
    p3 : np.ndarray
        Third vertex coordinates, shape (3,)
    normalized : bool, optional
        Whether to normalize the resulting vector, by default True

    Returns
    -------
    np.ndarray
        Normal vector, shape (3,). Normalized if requested.

    Raises
    ------
    ValueError
        If input points have invalid shape or are collinear
    """
    for point in (p1, p2, p3):
        if point.shape != (3,):
            raise ValueError("Points must be 3D coordinates")

    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)

    if np.linalg.norm(normal) < 1e-12:
        raise ValueError("Collinear points cannot form a valid normal vector")

    return normal / np.linalg.norm(normal) if normalized else normal


def get_element_type_from_numad(element: np.ndarray) -> ElementsTypes:
    """
    Determine element type from NuMAD's element connectivity array.

    Parameters
    ----------
    element : np.ndarray
        Element connectivity array with node indices

    Returns
    -------
    ElementsTypes
        Corresponding element type enum

    Raises
    ------
    NotImplementedError
        For unsupported element types
    """
    n_nodes = len(element)

    if n_nodes == 3:
        return ElementsTypes.TRIANGLE
    if n_nodes == 6:
        return ElementsTypes.TRIANGLE6
    if n_nodes == 4:
        return ElementsTypes.TRIANGLE if element[-1] < 0 else ElementsTypes.QUAD
    if n_nodes == 8:
        return ElementsTypes.TRIANGLE6 if element[-1] < 0 else ElementsTypes.QUAD8
    if n_nodes == 9:
        return ElementsTypes.QUAD9

    raise NotImplementedError(f"Element with {n_nodes} nodes not supported")


def pynumad_to_meshio(blade_mesh: PyNuMADMesh) -> meshio.Mesh:
    """
    Convert PyNuMAD mesh structure to meshio format.

    Parameters
    ----------
    blade_mesh : PyNuMADMesh
        Dictionary containing:
        - "nodes": Node coordinates array
        - "elements": Element connectivity list
        - "sets": Dictionary of element/node sets

    Returns
    -------
    meshio.Mesh
        Converted mesh with preserved sets

    Raises
    ------
    NotImplementedError
        For non-triangular elements
    """
    nodes = np.array(blade_mesh["nodes"])
    elements = np.array(blade_mesh["elements"])

    # Process element sets
    cell_sets = {s["name"]: [np.array(s["labels"])] for s in blade_mesh["sets"]["element"]}

    # Process node sets
    point_sets = {s["name"]: [np.array(s["labels"])] for s in blade_mesh["sets"]["node"]}

    # Organize elements by type
    cells: Dict[str, List[np.ndarray]] = {}
    for element in elements:
        elem_type = get_element_type_from_numad(element)

        if elem_type in (ElementsTypes.TRIANGLE, ElementsTypes.TRIANGLE6):
            key = "triangle6" if elem_type == ElementsTypes.TRIANGLE6 else "triangle"
            cells.setdefault(key, []).append(element)
        else:
            raise NotImplementedError("Only triangular elements currently supported")

    return meshio.Mesh(
        points=nodes,
        cells=[(k, np.array(v)) for k, v in cells.items()],
        cell_sets=cell_sets,
        point_sets=point_sets,
    )


def rotate_around_y(coords: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate 3D coordinates about the Y-axis.

    Parameters
    ----------
    coords : np.ndarray
        Input coordinates, shape (N, 3)
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray
        Rotated coordinates, shape (N, 3)
    """
    cosθ, sinθ = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cosθ, 0, sinθ], [0, 1, 0], [-sinθ, 0, cosθ]])
    return coords @ rotation_matrix.T


def array2str(header: str, matrix: np.ndarray, footer: str = "", max_size: int = 8) -> str:
    """
    Format a 2D array as a bordered table string with truncation.

    Parameters
    ----------
    header : str
        Table header text
    matrix : np.ndarray
        Input array to format
    footer : str, optional
        Footer text, by default ""
    max_size : int, optional
        Maximum number of rows/columns to show, by default 8

    Returns
    -------
    str
        Formatted table string with borders
    """
    matrix = np.array(matrix, dtype=object)
    nrows, ncols = matrix.shape
    cell_width = 14
    ellipsis_str = f"{'...':^{cell_width}}"

    def get_truncated_indices(total: int, max_size: int) -> list:
        if total <= max_size:
            return list(range(total)), []
        n_head = max_size // 2
        n_tail = max_size - n_head - 1
        return list(range(n_head)) + list(range(total - n_tail, total)), [n_head]

    row_idx, row_cuts = get_truncated_indices(nrows, max_size)
    col_idx, col_cuts = get_truncated_indices(ncols, max_size)

    truncated = matrix[np.ix_(row_idx, col_idx)]

    formatted = []
    for i, row in enumerate(truncated):
        if i in row_cuts:
            formatted.append([ellipsis_str] * len(col_idx))

        formatted_row = []
        for j, val in enumerate(row):
            if j in col_cuts:
                formatted_row.append(ellipsis_str)
            try:
                num = float(val)
                formatted_row.append(
                    f"{num:{cell_width}.3e}" if abs(num) > 1e-10 else " " * cell_width
                )
            except:
                formatted_row.append(f"{str(val):^{cell_width}}")
        formatted.append(formatted_row)

    ncols_final = len(formatted[0])
    border = "+" + "+".join(["-" * (cell_width + 2)] * ncols_final) + "+"
    h_border = border.replace("-", "=").replace(" ", "")

    table = [h_border, f"|{header:^{len(border) - 2}}|", h_border]
    for row in formatted:
        table.append("| " + " | ".join(row) + " |")
        table.append(border)
    table[-1] = h_border

    return "\n".join(table)


def quad_area(coords: np.ndarray) -> float:
    """
    Calculate area of quadrilateral elements.

    Supports 4-node (bilinear), 8-node (serendipity), and 9-node (Lagrange) elements.
    The 4-node element uses the shoelace formula, while the 8-node and 9-node elements
    are approximated by dividing them into 4 triangles.

    Parameters
    ----------
    coords : np.ndarray
        Node coordinates array with shape:
        - (4, 2) for bilinear quads
        - (8, 2) for serendipity quads
        - (9, 2) for Lagrange quads

    Returns
    -------
    float
        Calculated area

    Raises
    ------
    ValueError
        For invalid input shapes
    """
    if coords.shape not in [(4, 2), (8, 2), (9, 2)]:
        raise ValueError("Invalid element coordinates shape")

    if coords.shape == (4, 2):
        x, y = coords[:, 0], coords[:, 1]
        return 0.5 * abs(
            x[0] * y[1]
            + x[1] * y[2]
            + x[2] * y[3]
            + x[3] * y[0]
            - (y[0] * x[1] + y[1] * x[2] + y[2] * x[3] + y[3] * x[0])
        )

    elif coords.shape in [(8, 2), (9, 2)]:
        center = coords[8] if coords.shape == (9, 2) else coords[:4].mean(axis=0)

        triangles = [
            [coords[0], coords[1], center],
            [coords[1], coords[2], center],
            [coords[2], coords[3], center],
            [coords[3], coords[0], center],
        ]

        def triangle_area(p1, p2, p3):
            return 0.5 * abs(
                p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
            )

        return sum(triangle_area(*tri) for tri in triangles)
    else:
        raise ValueError("Input array must have shape (4,2), (8,2), or (9,2).")


def petsc_to_numpy(mat):
    """Converts a PETSc Mat to a NumPy array"""
    size = mat.getSize()
    np_mat = np.zeros(size)
    for i in range(size[0]):
        row_vals = mat.getValues(i, range(size[1]))
        np_mat[i, :] = row_vals
    return np_mat
