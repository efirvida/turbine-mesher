import meshio
import numpy as np

from .enums import Elements
from .types import PyNuMADMesh


def compute_triangular_element_normals(
    P1: np.ndarray, P2: np.ndarray, P3: np.ndarray, normalized: bool = True
) -> np.ndarray:
    """
    Calculates the normal vector of a triangle defined by three points in 3D space.

    Parameters:
    P1, P2, P3 : numpy.ndarray
        Arrays of shape (3,) representing the coordinates of the three vertices of the triangle
        (x, y, z).
    normalized : bool, optional
        If True (default), the function returns the normalized normal vector.
        If False, it returns the unnormalized normal vector.

    Returns:
    numpy.ndarray
        The calculated normal vector of the triangle. It will be normalized if `normalized` is True,
        otherwise, it will be unnormalized.

    Raises:
    ValueError
        If the input points are not numpy arrays of shape (3,) or if the points are collinear (in which case,
        the normal vector cannot be calculated).

    Notes:
    The function computes the normal vector using the cross product of two vectors defined by the points.
    The normal is only valid if the points are not collinear.
    """
    for point in [P1, P2, P3]:
        if not isinstance(point, np.ndarray) or point.shape != (3,):
            raise ValueError("Each point must be a numpy array of shape (3,)")

    v1 = P2 - P1
    v2 = P3 - P1

    normal = np.cross(v1, v2)

    # Normalize the normal if required
    if normalized:
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        else:
            raise ValueError("The points are collinear; normal cannot be calculated.")

    return normal


def get_element_type_from_numad(element):
    if len(element) == 3:
        return Elements.TRIANGLE
    if len(element) == 6:
        return Elements.TRIANGLE6
    if len(element) == 4:
        if element[-1] < 0:
            return Elements.TRIANGLE
        else:
            return Elements.QUAD
    elif len(element) == 8:
        if element[-1] < 0:
            return Elements.TRIANGLE6
        else:
            return Elements.QUAD8
    else:
        raise NotImplementedError(
            "Only supported triangle, 6node triangles, quad and 8 node quad elements"
        )


# FIXME
def pynumad_to_meshio(blade_mesh: PyNuMADMesh) -> meshio.Mesh:
    """
    Convert a PyNuMAD mesh to a `meshio.Mesh` object.

    Parameters:
    blade_mesh : dict
        A dictionary containing the PyNuMAD mesh data with keys such as "nodes", "elements", and "sets".

    Returns:
    meshio.Mesh
        A `meshio.Mesh` object created from the PyNuMAD mesh data, including nodes, elements, cell sets, and point sets.

    Notes:
    - The function assumes that the elements are either triangles or 6-node triangles.
    - The function raises a NotImplementedError if an element type other than triangle or 6-node triangle is encountered.
    """
    nodes = np.array(blade_mesh["nodes"])
    elements = np.array(blade_mesh["elements"])

    # Create cell_sets and point_sets
    cell_sets = {i["name"]: [np.array(i["labels"])] for i in blade_mesh["sets"]["element"]}
    point_sets = {i["name"]: [np.array(i["labels"])] for i in blade_mesh["sets"]["node"]}

    cells = {}
    for element in elements:
        if len(element) == 3:
            if "triangle" not in cells:
                cells["triangle"] = []
            cells["triangle"].append(element)
        elif len(element) == 6:
            if "triangle6" not in cells:
                cells["triangle6"] = []
            cells["triangle6"].append(element)
        else:
            raise NotImplementedError(
                "Only triangles (3 nodes) and triangle6 (6 nodes) are supported."
            )

    return meshio.Mesh(points=nodes, cells=cells, cell_sets=cell_sets, point_sets=point_sets)


def rotate_around_y(coords, angle) -> np.ndarray:
    """
    Rotate the mesh around the Y axis by a given angle.

    :param mesh: The mesh to rotate. It should be a numpy array of shape (N, 3).
    :param angle: The angle in radians to rotate the mesh.
    :return: The rotated mesh.
    """
    # Rotation matrix around the Y axis
    rotation_matrix = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )

    # Apply the rotation to each node in the mesh
    rotated_mesh = np.dot(coords, rotation_matrix.T)

    return rotated_mesh.copy()
