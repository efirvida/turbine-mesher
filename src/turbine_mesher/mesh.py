import os
from copy import deepcopy
from typing import Dict, Tuple

import meshio
import numpy as np
import pynumad as pynu
import trimesh
from pynumad.mesh_gen.mesh_gen import shell_mesh_general, solidMeshFromShell

from turbine_mesher.io.yaml import read_hub_data


class Mesh:
    def __init__(
        self,
        numad_yaml: str,
        element_size: float = 0.5,
        n_samples: int = 300,
    ):
        self._yaml = numad_yaml
        self._blade = pynu.Blade()
        self._blade.read_yaml(numad_yaml)
        self._hub = read_hub_data(numad_yaml)

        self._mesh_element_size = element_size
        self._layer_num_els = (1, 1, 1)

        self.has_shell_mesh = False
        self.has_solid_mesh = False

        for stat in self._blade.definition.stations:
            stat.airfoil.resample(n_samples=n_samples)

        self._blade.update_blade()
        nStations = self._blade.geometry.coordinates.shape[2]
        minTELengths = 0.001 * np.ones(nStations)
        self._blade.expand_blade_geometry_te(minTELengths)

    def shell_mesh(self):
        print("#######################")
        print("# Creating Shell Mesh #")
        print("#######################")
        mesh = shell_mesh_general(self._blade, False, False, self._mesh_element_size)
        print("Meshing Done!")
        print("#######################")

        self.blade_root_nodes = np.array([
            mesh["nodes"][node] for node in mesh["sets"]["node"][0]["labels"]
        ])

        surface_elements_ids = [
            label
            for element in mesh["sets"]["element"]
            if "w" not in element["name"].lower()
            for label in element["labels"]
        ]

        surface_elements = [mesh["elements"][el] for el in surface_elements_ids]
        self.surface_nodes = {
            int(node): mesh["nodes"][node]
            for node in np.array(surface_elements).flatten()
            if node != -1
        }
        print("Meshing Done!")
        print("#######################")
        self.pynumad_shell_mesh = mesh
        self.has_shell_mesh = True
        self.has_solid_mesh = False
        self._pynumad_to_meshio(self.pynumad_shell_mesh)

    def solid_mesh(self):
        print("#######################")
        print("# Creating Solid Mesh #")
        print("#######################")
        self._blade.stackdb.edit_stacks_for_solid_mesh()
        shell_mesh = shell_mesh_general(self._blade, True, True, self._mesh_element_size)
        print("finished shell mesh")
        mesh = solidMeshFromShell(
            self._blade, shell_mesh, self._layer_num_els, self._mesh_element_size
        )
        print("Meshing Done!")
        print("#######################")

        self.blade_root_nodes = [mesh["nodes"][node] for node in mesh["sets"]["node"][0]["labels"]]
        self.pynumad_solid_mesh = mesh
        self.has_shell_mesh = False
        self.has_solid_mesh = True

        self._pynumad_to_meshio(self.pynumad_solid_mesh)

    def mesh_rotor(self, n_blades: int = 3):
        """
        Generate a mesh for the complete rotor with n_blade blades.

        :param blade_mesh: Single blade mesh object (meshio.Mesh).
        :param n_blade: Number of blades to generate.
        :param hub_radius: Radius of the hub.
        :return: Full rotor mesh (meshio.Mesh).
        """
        blade_mesh = deepcopy(self.mesh)
        blade_mesh.points[:, 2] += self._hub.radius

        rotor_points = blade_mesh.points.copy()

        # Maintain the original node count
        total_points = rotor_points.copy()
        total_cells = []  # Empty list to store the cells

        # Renumber elements for each blade and add them to the rotor
        for i in range(0, n_blades):
            angle = 360 * i / n_blades  # Rotation angle for this blade
            rotated_mesh = rotate_mesh(blade_mesh, angle, axis="y")

            # Renumber nodes and elements
            offset = len(total_points)  # Offset is the number of existing nodes
            rotated_points = rotated_mesh.points
            rotated_cells = rotated_mesh.cells

            # Renumber nodes: adjust the indices of elements
            total_points = np.vstack([total_points, rotated_points])

            # Renumber elements: adjust the node indices in each cell
            for cell in rotated_cells:  # Iterate over each cell (type, nodes)
                cell_type = cell.type
                cell_nodes = cell.data
                new_cell_nodes = cell_nodes + offset
                total_cells.append((cell_type, new_cell_nodes))

        # Create a new mesh with renumbered points and cells
        self.mesh = meshio.Mesh(total_points, total_cells)

    def write(self, output_file: str, **kwargs):
        """
        Writes a solid mesh to a file, creating directories if needed.

        :param mesh_obj: Mesh object (MeshIO or PyNuMAD).
        :param file_name: File name to write the mesh.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Determine if remeshing is needed based on file extension
        file_extension = output_file.lower().split(".")[-1]
        mesh = deepcopy(self.mesh)
        if file_extension in {"stl", "obj"}:
            mesh = self._remesh_with_trimesh(mesh)

        # Write using meshio for other formats
        meshio.write(output_file, mesh, **kwargs)

    def _meshio_to_trimesh(self):
        """
        Convert a meshio Mesh to a trimesh object with surface data (triangular faces).

        :param meshio_mesh: meshio Mesh object (can contain hexahedrons, wedges, quads, and triangles).
        :return: trimesh.Trimesh object (surface only).
        """
        # Extract points and cell data
        points = self.mesh.points
        cells = self.mesh.cells_dict

        faces = []

        # Handle hexahedrons (C3D8) - Generate triangular faces
        for cell in cells.get("hexahedron", []):
            hex_nodes = cell
            faces.extend([
                [hex_nodes[0], hex_nodes[1], hex_nodes[2]],
                [hex_nodes[0], hex_nodes[2], hex_nodes[3]],
                [hex_nodes[0], hex_nodes[3], hex_nodes[7]],
                [hex_nodes[0], hex_nodes[7], hex_nodes[4]],
                [hex_nodes[4], hex_nodes[5], hex_nodes[6]],
                [hex_nodes[4], hex_nodes[6], hex_nodes[7]],
                [hex_nodes[2], hex_nodes[3], hex_nodes[6]],
                [hex_nodes[3], hex_nodes[7], hex_nodes[6]],
                [hex_nodes[1], hex_nodes[2], hex_nodes[5]],
                [hex_nodes[2], hex_nodes[6], hex_nodes[5]],
                [hex_nodes[1], hex_nodes[5], hex_nodes[4]],
                [hex_nodes[1], hex_nodes[4], hex_nodes[0]],
            ])

        # Handle wedges (C3D6) - Generate triangular faces
        for cell in cells.get("wedge", []):
            wedge_nodes = cell
            faces.extend([
                [wedge_nodes[0], wedge_nodes[1], wedge_nodes[2]],
                [wedge_nodes[0], wedge_nodes[2], wedge_nodes[3]],
                [wedge_nodes[0], wedge_nodes[3], wedge_nodes[4]],
                [wedge_nodes[0], wedge_nodes[4], wedge_nodes[5]],
                [wedge_nodes[1], wedge_nodes[2], wedge_nodes[3]],
                [wedge_nodes[1], wedge_nodes[3], wedge_nodes[4]],
                [wedge_nodes[1], wedge_nodes[4], wedge_nodes[5]],
            ])

        # Handle shell quads - Split quads into two triangles
        for cell in cells.get("quad", []):
            quad_nodes = cell
            faces.extend([
                [quad_nodes[0], quad_nodes[1], quad_nodes[2]],
                [quad_nodes[0], quad_nodes[2], quad_nodes[3]],
            ])

        # Handle shell triangles - Use directly as faces
        for cell in cells.get("triangle", []):
            tri_nodes = cell
            faces.append(tri_nodes)

        # Convert faces to trimesh format
        trimesh_mesh = trimesh.Trimesh(vertices=points, faces=np.array(faces))

        return trimesh_mesh

    def _remesh_with_trimesh(self):
        """
        Remesh the mesh using trimesh (surface remeshing for hexahedrons and wedges).

        :param meshio_mesh: meshio Mesh object.
        :return: remeshed meshio Mesh object.
        """
        # Convert meshio to trimesh (surface)
        trimesh_mesh = self._meshio_to_trimesh(self.mesh)

        # Perform remeshing on the surface (subdivide or smooth)
        remeshed_surface = trimesh_mesh.subdivide()  # Example: subdivision

        # Convert back to meshio format (using the remeshed surface)
        remeshed_mesh = meshio.Mesh(
            points=remeshed_surface.vertices, cells=[("triangle", remeshed_surface.faces)]
        )

        return remeshed_mesh

    def _pynumad_to_meshio(self, blade_mesh) -> meshio.Mesh:
        """
        Convert a PyNuMAD blade mesh to a `meshio.Mesh` object and optionally save it to a file.

        :param blade_mesh: Dictionary representing the PyNuMAD blade mesh.
        :param file_name: Optional file name to save the converted mesh.
        :return: `meshio.Mesh` object.
        """
        nodes = np.array(blade_mesh["nodes"])
        elements = np.array(blade_mesh["elements"])

        # Clasificación de celdas según las reglas dadas
        cells = {"quad": [], "triangle": [], "hexahedron": [], "wedge": []}
        for element in elements:
            if len(element) == 4 and element[3] != -1:  # Quad (S4)
                cells["quad"].append(element[:4])
            elif len(element) == 4 and element[3] == -1:  # Triangle (S3)
                cells["triangle"].append(element[:3])
            elif len(element) == 8 and element[6] != -1:  # Hexahedron (C3D8I)
                cells["hexahedron"].append(element[:8])
            elif len(element) == 8 and element[6] == -1:  # Wedge (C3D6)
                cells["wedge"].append(element[:6])
            else:
                raise ValueError(f"Unknown element type with data: {element}")

        # Convertir las listas de celdas en formato `meshio`
        mesh_cells = [(key, np.array(value)) for key, value in cells.items() if value]
        self.mesh = meshio.Mesh(points=nodes, cells=mesh_cells)

        nodes_dim_tags = [[0, n] for n in range(nodes.shape[0])]
        # elements_dim_tags = [[2, e] for e in range(elements.max())]
        dim_tags = {"gmsh:dim_tags": nodes_dim_tags}

        cell_data = {"gmsh:geometrical": [(2, i) for i in range(len(elements))]}

        self.mesh.point_data = dim_tags
        self.mesh.cell_data = cell_data

    def to_gmsh(self):
        mesh = deepcopy(self.pynumad_solid_mesh)
        nodes = mesh["nodes"]
        elements = mesh["elements"]

        mesh_str = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes"]
        mesh_str.append(f"{nodes.shape[0]}")
        mesh_str.extend([f"{i} {n[0]} {n[1]} {n[2]}" for i, n in enumerate(nodes)])
        mesh_str.append("$EndNodes")

        mesh_str.append("$Elements")
        mesh_str.append(f"{elements.shape[0]}")
        for el_id, element in enumerate(elements):
            if len(element) == 8 and element[6] != -1:  # Hexahedron (C3D8I)
                el_type = 5
            elif len(element) == 8 and element[6] == -1:  # Wedge (C3D6)
                el_type = 6

            nodes = " ".join(str(el) for el in element if el != -1)

            mesh_str.append(f"{el_id} {el_type} 0 {nodes}")

        mesh_str.append("$EndElements")

        with open("gmsh.msh", "w") as f:
            f.write("\n".join(mesh_str))

    def __str__(self):
        # Inicializar una lista para almacenar las líneas del reporte
        report = ["Mesh statistics:"]

        # Agregar el número total de puntos
        report.append(f"Total nodes: {self.mesh.points.shape[0]}")

        # Calcular y agregar el número total de celdas
        total_cells = sum(cell.data.shape[0] for cell in self.mesh.cells)
        report.append(f"Total cells: {total_cells}")

        # Calcular y agregar los tipos de celdas y sus conteos
        cell_info = {t.type: 0 for t in self.mesh.cells}
        for cell in self.mesh.cells:
            amount = cell_info[cell.type] + cell.data.shape[0]
            cell_info.update({cell.type: amount})

        report.append(f"Cell types: {', '.join(cell_info.keys())}")
        report.extend([f" -> {type}: {number} Cells" for type, number in cell_info.items()])

        # Unir las líneas del reporte y devolverlo como una cadena
        return "\n".join(report)


def solid_mesh(blade, elementSize: float, layerNumEls: Tuple[int, int, int] = (1, 1, 1)):
    print("#######################")
    print("# Creating Solid Mesh #")
    print("#######################")
    blade.stackdb.edit_stacks_for_solid_mesh()
    shellMesh = shell_mesh_general(blade, True, True, elementSize)
    print("finished shell mesh")
    mesh = solidMeshFromShell(blade, shellMesh, layerNumEls, elementSize)
    print("Meshing Done!")
    print("#######################")
    root_nodes = [mesh["nodes"][node] for node in mesh["sets"]["node"][0]["labels"]]
    np.savetxt("/home/infralab/Desktop/turbine-mesher/rootNodes.txt", root_nodes)
    return pynumad_to_meshio(mesh)


def shell_mesh(blade, elementSize: float):
    print("#######################")
    print("# Creating Shell Mesh #")
    print("#######################")
    mesh = shell_mesh_general(blade, False, False, elementSize)
    print("Meshing Done!")
    print("#######################")
    root_nodes = [mesh["nodes"][node] for node in mesh["sets"]["node"][0]["labels"]]
    surface_elements_ids = [
        element["labels"]
        for element in mesh["sets"]["element"]
        if "w" not in element["name"].lower()
    ]
    surface_elements_ids_flat = []
    [surface_elements_ids_flat.extend(a) for a in surface_elements_ids]
    surface_elements = [mesh["elements"][el] for el in surface_elements_ids_flat]
    surface_nodes = [
        mesh["nodes"][node] for node in np.array(surface_elements).flatten() if node != -1
    ]
    np.savetxt("/home/infralab/Desktop/turbine-mesher/nodes.txt", surface_nodes)

    return pynumad_to_meshio(mesh)


def meshio_to_trimesh(meshio_mesh):
    """
    Convert a meshio Mesh to a trimesh object with surface data (triangular faces).

    :param meshio_mesh: meshio Mesh object (can contain hexahedrons, wedges, quads, and triangles).
    :return: trimesh.Trimesh object (surface only).
    """
    # Extract points and cell data
    points = meshio_mesh.points
    cells = meshio_mesh.cells_dict

    faces = []

    # Handle hexahedrons (C3D8) - Generate triangular faces
    for cell in cells.get("hexahedron", []):
        hex_nodes = cell
        faces.extend([
            [hex_nodes[0], hex_nodes[1], hex_nodes[2]],
            [hex_nodes[0], hex_nodes[2], hex_nodes[3]],
            [hex_nodes[0], hex_nodes[3], hex_nodes[7]],
            [hex_nodes[0], hex_nodes[7], hex_nodes[4]],
            [hex_nodes[4], hex_nodes[5], hex_nodes[6]],
            [hex_nodes[4], hex_nodes[6], hex_nodes[7]],
            [hex_nodes[2], hex_nodes[3], hex_nodes[6]],
            [hex_nodes[3], hex_nodes[7], hex_nodes[6]],
            [hex_nodes[1], hex_nodes[2], hex_nodes[5]],
            [hex_nodes[2], hex_nodes[6], hex_nodes[5]],
            [hex_nodes[1], hex_nodes[5], hex_nodes[4]],
            [hex_nodes[1], hex_nodes[4], hex_nodes[0]],
        ])

    # Handle wedges (C3D6) - Generate triangular faces
    for cell in cells.get("wedge", []):
        wedge_nodes = cell
        faces.extend([
            [wedge_nodes[0], wedge_nodes[1], wedge_nodes[2]],
            [wedge_nodes[0], wedge_nodes[2], wedge_nodes[3]],
            [wedge_nodes[0], wedge_nodes[3], wedge_nodes[4]],
            [wedge_nodes[0], wedge_nodes[4], wedge_nodes[5]],
            [wedge_nodes[1], wedge_nodes[2], wedge_nodes[3]],
            [wedge_nodes[1], wedge_nodes[3], wedge_nodes[4]],
            [wedge_nodes[1], wedge_nodes[4], wedge_nodes[5]],
        ])

    # Handle shell quads - Split quads into two triangles
    for cell in cells.get("quad", []):
        quad_nodes = cell
        faces.extend([
            [quad_nodes[0], quad_nodes[1], quad_nodes[2]],
            [quad_nodes[0], quad_nodes[2], quad_nodes[3]],
        ])

    # Handle shell triangles - Use directly as faces
    for cell in cells.get("triangle", []):
        tri_nodes = cell
        faces.append(tri_nodes)

    # Convert faces to trimesh format
    trimesh_mesh = trimesh.Trimesh(vertices=points, faces=np.array(faces))

    return trimesh_mesh


def remesh_with_trimesh(meshio_mesh):
    """
    Remesh the mesh using trimesh (surface remeshing for hexahedrons and wedges).

    :param meshio_mesh: meshio Mesh object.
    :return: remeshed meshio Mesh object.
    """
    # Convert meshio to trimesh (surface)
    trimesh_mesh = meshio_to_trimesh(meshio_mesh)

    # Perform remeshing on the surface (subdivide or smooth)
    remeshed_surface = trimesh_mesh.subdivide()  # Example: subdivision

    # Convert back to meshio format (using the remeshed surface)
    remeshed_mesh = meshio.Mesh(
        points=remeshed_surface.vertices, cells=[("triangle", remeshed_surface.faces)]
    )

    return remeshed_mesh


def mesh_statistics(mesh):
    """
    Print general statistics of the mesh.

    :param mesh: Mesh object (meshio.Mesh).
    """
    print("Mesh statistics:")

    # Total number of points
    print(f"Total nodes: {mesh.points.shape[0]}")

    # Total number of cells
    total_cells = sum([cell.data.shape[0] for cell in mesh.cells])
    print(f"Total cells: {total_cells}")

    # Types of cells and how many there are of each type
    cell_types = set(cell.type for cell in mesh.cells)
    print(f"Cell types: {cell_types}")
    for cell_type in cell_types:
        cell_count = sum([1 for cell in mesh.cells if cell.type == cell_type])
        print(f"  {cell_type}: {cell_count} cells")


def pynumad_to_meshio(blade_mesh: Dict, file_name: str = None) -> meshio.Mesh:
    """
    Convert a PyNuMAD blade mesh to a `meshio.Mesh` object and optionally save it to a file.

    :param blade_mesh: Dictionary representing the PyNuMAD blade mesh.
    :param file_name: Optional file name to save the converted mesh.
    :return: `meshio.Mesh` object.
    """
    nodes = np.array(blade_mesh["nodes"])
    elements = np.array(blade_mesh["elements"])

    # Clasificación de celdas según las reglas dadas
    cells = {"quad": [], "triangle": [], "hexahedron": [], "wedge": []}

    for element in elements:
        if len(element) == 4 and element[3] != -1:  # Quad (S4)
            cells["quad"].append(element[:4])
        elif len(element) == 4 and element[3] == -1:  # Triangle (S3)
            cells["triangle"].append(element[:3])
        elif len(element) == 8 and element[6] != -1:  # Hexahedron (C3D8I)
            cells["hexahedron"].append(element[:8])
        elif len(element) == 8 and element[6] == -1:  # Wedge (C3D6)
            cells["wedge"].append(element[:6])
        else:
            raise ValueError(f"Unknown element type with data: {element}")

    # Convertir las listas de celdas en formato `meshio`
    mesh_cells = [(key, np.array(value)) for key, value in cells.items() if value]

    # Crear el objeto meshio.Mesh
    mesh = meshio.Mesh(points=nodes, cells=mesh_cells)

    # Escribir el archivo opcionalmente
    if file_name:
        meshio.write(file_name, mesh)

    return mesh


def rotate_mesh(mesh, angle_deg, axis="z"):
    """
    Rotate the mesh around a given axis (default is 'z').

    :param mesh: Mesh object (meshio.Mesh).
    :param angle_deg: Rotation angle in degrees.
    :param axis: Axis around which to rotate ('x', 'y', 'z').
    :return: New rotated mesh.
    """
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.eye(3)

    if axis == "x":
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)],
        ])
    elif axis == "y":
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)],
        ])
    elif axis == "z":
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1],
        ])

    # Apply rotation to each node in the mesh
    rotated_points = np.dot(mesh.points, rotation_matrix.T)
    return meshio.Mesh(rotated_points, mesh.cells)


def mesh_rotor(blade_mesh, n_blade, hub_radius):
    """
    Generate a mesh for the complete rotor with n_blade blades.

    :param blade_mesh: Single blade mesh object (meshio.Mesh).
    :param n_blade: Number of blades to generate.
    :param hub_radius: Radius of the hub.
    :return: Full rotor mesh (meshio.Mesh).
    """
    if isinstance(blade_mesh, dict):
        blade_mesh = pynumad_to_meshio(blade_mesh)
    blade_mesh.points[:, 2] += hub_radius

    rotor_points = blade_mesh.points.copy()

    # Maintain the original node count
    total_points = rotor_points.copy()
    total_cells = []  # Empty list to store the cells

    # Renumber elements for each blade and add them to the rotor
    for i in range(0, n_blade):
        angle = 360 * i / n_blade  # Rotation angle for this blade
        rotated_mesh = rotate_mesh(blade_mesh, angle, axis="y")

        # Renumber nodes and elements
        offset = len(total_points)  # Offset is the number of existing nodes
        rotated_points = rotated_mesh.points
        rotated_cells = rotated_mesh.cells

        # Renumber nodes: adjust the indices of elements
        total_points = np.vstack([total_points, rotated_points])

        # Renumber elements: adjust the node indices in each cell
        for cell in rotated_cells:  # Iterate over each cell (type, nodes)
            cell_type = cell.type
            cell_nodes = cell.data
            new_cell_nodes = cell_nodes + offset
            total_cells.append((cell_type, new_cell_nodes))

    # Create a new mesh with renumbered points and cells
    rotor_mesh = meshio.Mesh(total_points, total_cells)

    return rotor_mesh
