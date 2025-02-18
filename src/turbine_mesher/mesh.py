import os
from typing import Dict, List, Self, Set

import numpy as np
import pyvista as pv
from rich.progress import Progress
from scipy.spatial import KDTree

from .enums import ELEMENTS_TO_CALCULIX, ELEMENTS_TO_VTK, Elements
from .helpers import get_element_type_from_numad
from .types import PyNuMADMesh
from .viewer import plot_mesh


class BaseMesh:
    def __init__(
        self,
        use_quadratic_elements: bool = True,
        enforce_triangular_elements: bool = False,
    ) -> None:
        self._qudratic_elements = use_quadratic_elements
        self._enforce_triangular_elements = enforce_triangular_elements

        self._mesh = {
            "nodes": [],
            "elements": [],
            "sets": {"node": [], "element": []},
            "materials": [],
            "sections": [],
        }

    def shell_mesh(self) -> Self:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def mesh(self) -> PyNuMADMesh:
        """
        Retrieves the mesh object for the blade, generating it if it does not exist.

        Returns:
        --------
        PyNuMADMesh
            The mesh object representing the discretized geometry of the blade, either created on demand or retrieved
            from the cached value.

        Notes:
        ------
        This property checks if a mesh has already been generated (stored in `_mesh`). If not, it triggers the creation
        of the mesh using the `__shell_mesh` method. The mesh may be generated using linear or quadratic elements depending
        on the configuration provided to the Blade object.
        """
        if len(self._mesh["nodes"]):
            return self._mesh
        self.shell_mesh()

    @property
    def num_nodes(self):
        return self.nodes.shape[0]

    @property
    def x(self):
        return self.nodes[:, 0]

    @property
    def y(self):
        return self.nodes[:, 1]

    @property
    def z(self):
        return self.nodes[:, 2]

    @property
    def nodes(self) -> np.ndarray:
        return np.array(self.mesh["nodes"], dtype=np.float64)

    @property
    def elements(self) -> np.ndarray:
        return np.array(self.mesh["elements"], dtype=np.int32)

    @property
    def element_sets(self) -> Dict[str, List[int]]:
        return {el_set["name"]: el_set["labels"] for el_set in self.mesh["sets"]["element"]}

    @property
    def node_sets(self) -> Dict[str, List[int]]:
        return {node_set["name"]: node_set["labels"] for node_set in self.mesh["sets"]["node"]}

    def get_element_sets(self, element_id) -> Set:
        return {
            set_name for set_name, elements in self.element_sets.items() if element_id in elements
        }

    def get_node_sets(self, node_id) -> Set:
        return {set_name for set_name, nodes in self.node_sets.items() if node_id in nodes}

    @property
    def elements_class(self) -> Dict[Elements, List[int]]:
        elements_map = {el_type: [] for el_type in ELEMENTS_TO_CALCULIX}
        for el_id, element in enumerate(self.elements):
            el_type = get_element_type_from_numad(element)
            if el_type in elements_map:
                elements_map[el_type].append(el_id)
        return elements_map

    def write_mesh(self, output_file: str) -> None:
        """
        Writes the mesh data to a specified output file in CalculiX (INP) or VTK format.

        Parameters:
        -----------
        output_file : str
            Path to the output file where the mesh will be saved. The file extension (e.g., '.inp' or '.vtk')
            should correspond to the desired format.

        Returns:
        --------
        None
            This function does not return any value. It saves the mesh data to the specified output file.

        Raises:
        -------
        NotImplementedError
            If an unsupported format (other than "inp" or "vtk") is provided, the function raises a NotImplementedError.

        Notes:
        ------
        This function automatically checks the specified file format and calls the appropriate helper function
        (`__write_inp` for CalculiX input files or `__write_vtk` for VTK files) to save the mesh data.
        If the provided `output_file` does not exist, the function will create the necessary directories before
        saving the file.
        """

        _ = self.mesh  # <- ensure meshing

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if output_file.lower().endswith("inp"):
            self.__write_inp(output_file)
        elif output_file.lower().endswith("vtk"):
            self.__write_vtk(output_file)
        else:
            raise NotImplementedError(
                "Only supported formats are 'inp' (CalculiX) and 'vtk' (VTK)."
            )

    def __write_inp(self, filename: str) -> None:
        """
        Writes the mesh data to a CalculiX input (INP) file.

        Parameters:
        -----------
        filename : str
            The name (including path) of the INP file to be created. This file will contain the mesh data in the
            CalculiX input format, which can be used for finite element analysis simulations.

        mesh : PyNuMADMesh
            The mesh object containing the mesh data to be written to the INP file. The mesh data should include the
            necessary element types, node information, and material properties as required by the CalculiX solver.

        Returns:
        --------
        None
            This function does not return any value. It writes the mesh data to the specified INP file.

        Notes:
        ------
        The function will translate the mesh data, including elements, nodes, and other necessary details, into a format
        compatible with CalculiX input files. It ensures the proper formatting and organization of the mesh data,
        allowing it to be used directly in a CalculiX simulation.

        This method is typically used when preparing mesh data for structural or mechanical simulations using the
        CalculiX solver, which requires input files in the INP format.

        Example usage:
        --------------
        To write a mesh to a CalculiX input file, use the following:

        ```python
        blade = Blade(yaml_file="path/to/blade_config.yaml")
        blade.write_mesh("output_mesh.inp", format="inp")
        ```

        The function automatically formats the mesh data and writes it to the specified file.
        """

        def split(arr):
            chunk_size = 8
            return [arr[i : i + chunk_size] for i in range(0, len(arr), chunk_size)]

        with open(filename, "wt") as f:
            f.write("**************++***********\n")
            f.write("**      MESH NODES       **\n")
            f.write("***************************\n")
            f.write("*NODE, NSET=N_ALL\n")
            for i, nd in enumerate(self.nodes, start=1):
                ln = f"{str(i)}, {', '.join(f'{n:.10e}' for n in nd)} \n"
                f.write(ln)

            f.write("\n**************++**++++********\n")
            f.write("**      MESH ELEMENTS       **\n")
            f.write("***********************+++****\n")
            for element_kind in ELEMENTS_TO_CALCULIX:
                if self.elements_class[element_kind]:
                    f.write(
                        f"*ELEMENT, TYPE={ELEMENTS_TO_CALCULIX[element_kind].value}, ELSET=E_ALL\n"
                    )
                    for i, el in enumerate(self.elements, start=1):
                        if element_kind == get_element_type_from_numad(el):
                            ln = f"{str(i)}, {', '.join(str(n + 1) for n in el if n != -1)} \n"
                            f.write(ln)

            f.write("\n**************+++*****+**********++**************\n")
            f.write("**      ELEMENT SETS DEFINITION SECTION       **\n")
            f.write("*******************+++++************************\n")

            for name, labels in self.element_sets.items():
                ln = f"*ELSET, ELSET=E_{name}\n"
                f.write(ln)
                for el in split(labels):
                    ln = ", ".join(str(e + 1) for e in el) + "\n"
                    f.write(ln)
                f.write("\n")

            f.write("\n**************************+****++*************\n")
            f.write("**       NODE SETS DEFINITION SECTION       **\n")
            f.write("*************************+++******************\n")

            for name, labels in self.node_sets.items():
                ln = f"*NSET, NSET=N_{name}\n"
                f.write(ln)
                for nd in split(labels):
                    ln = ", ".join(str(n + 1) for n in nd) + "\n"
                    f.write(ln)
                f.write("\n")

            f.write("\n*******************************************\n")
            f.write("**            MATERIALS SECTION          **\n")
            f.write("*******************************************\n")

            for mat in self.mesh["materials"]:
                ln = "*MATERIAL, NAME=" + mat["name"] + "\n"
                f.write(ln)
                f.write("*DENSITY\n")
                ln = str(mat["density"]) + ",\n"
                f.write(ln)
                f.write("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n")
                E = mat["elastic"]["E"]
                nu = mat["elastic"]["nu"]
                G = mat["elastic"]["G"]
                eProps = [str(E[0]), str(E[1]), str(E[2])]
                eProps.extend([str(nu[0]), str(nu[1]), str(nu[2])])
                eProps.extend([str(G[0]), str(G[1]), str(G[2])])
                ln = ", ".join(eProps[0:8]) + "\n"
                f.write(ln)
                ln = eProps[8] + ",\n\n"
                f.write(ln)

            # f.write("\n**************************************\n")
            # f.write("**        ORIENTATION SECTION       **\n")
            # f.write("**************************************\n")
            # for sec in mesh["sections"]:
            #     ln = "*ORIENTATION, NAME=ORI_" + sec["elementSet"] + ", SYSTEM=RECTANGULAR\n"
            #     f.write(ln)
            #     dataLn = list()
            #     for d in sec["xDir"]:
            #         dataLn.append(f"{d:.10e}")
            #     for d in sec["xyDir"]:
            #         dataLn.append(f"{d:.10e}")
            #     dataStr = ", ".join(dataLn) + "\n"
            #     f.write(dataStr)
            #     f.write("3, 0.\n\n")

            f.write("\n*******************************************\n")
            f.write("**        SHELL DEFINITION SECTION       **\n")
            f.write("*******************************************\n")
            if True:
                for sec in self.mesh["sections"]:
                    snm = sec["elementSet"]
                    # ln = f"*SHELL SECTION, ELSET=E_{snm}, ORIENTATION=ORI_{snm}, MATERIAL={sec['layup'][0][0]}, OFFSET=0.0\n"
                    ln = f"*SHELL SECTION, ELSET=E_{snm}, MATERIAL={sec['layup'][0][0]}, OFFSET=0.0\n"
                    f.write(ln)
                    thikness = sum(lay[1] for lay in sec["layup"])
                    f.write(f"{thikness:.10e}\n")
            else:
                for sec in mesh["sections"]:
                    snm = sec["elementSet"]
                    ln = f"*SHELL SECTION, ELSET=E_{snm}, COMPOSITE, ORIENTATION=ORI_{snm}, OFFSET=0.0\n"
                    f.write(ln)
                    for lay in sec["layup"]:
                        layStr = f"{lay[1]:.10e}, {lay[2]:.10e}, {lay[0]}\n"
                        f.write(layStr)
                    f.write("\n")

            f.write("\n********************************\n")
            f.write("**        STEPS SECTION       **\n")
            f.write("********************************\n")

    def __write_vtk(self, filename: str) -> None:
        """
        Writes the mesh data to a VTK (Visualization Toolkit) file format.

        Parameters:
        -----------
        filename : str
            The name (including path) of the VTK file to be created. This file will contain the mesh data
            in the VTK format, which is commonly used for visualization and post-processing purposes.

        mesh : PyNuMADMesh
            The mesh object containing the mesh data to be written to the VTK file. The mesh should include
            node coordinates, element connectivity, and any other necessary attributes required for visualization.

        Returns:
        --------
        None
            This function does not return any value. It writes the mesh data to the specified VTK file.

        Notes:
        ------
        The VTK file format is widely used for visualizing mesh data in post-processing tools such as ParaView,
        VisIt, or other visualization software. This function translates the mesh data into the VTK format,
        ensuring it is compatible with these tools.

        The VTK format supports a variety of visualization features, and the output file will be readable by any
        software that supports VTK files, allowing for mesh rendering and further analysis.

        Example usage:
        --------------
        To write a mesh to a VTK file, you can use the following:

        ```python
        blade = Blade(yaml_file="path/to/blade_config.yaml")
        blade.write_mesh("output_mesh.vtk", format="vtk")
        ```

        This will create a VTK file containing the mesh data, which can then be loaded into a visualization
        software for inspection.
        """
        vtk_to_numpy_dtype_name = {
            "float": "float32",
            "double": "float64",
            "int": "int",
            "vtktypeint8": "int8",
            "vtktypeint16": "int16",
            "vtktypeint32": "int32",
            "vtktypeint64": "int64",
            "vtktypeuint8": "uint8",
            "vtktypeuint16": "uint16",
            "vtktypeuint32": "uint32",
            "vtktypeuint64": "uint64",
        }
        numpy_to_vtk_dtype = {v: k for k, v in vtk_to_numpy_dtype_name.items()}

        def _write_points(f, points):
            dtype = numpy_to_vtk_dtype[points.dtype.name]
            f.write(f"POINTS {len(points)} {dtype}\n")
            for point in points:
                f.write(f"{' '.join(f'{p:.5f}' for p in point)}\n")
            f.write("\n")

        def _write_cells(f, elements):
            total_size = len(self.elements) + sum(1 for el in self.elements for n in el if n != -1)

            f.write(f"CELLS {elements.shape[0]} {total_size}\n")
            for cell in elements:
                nodes = [str(c) for c in cell if c != -1]
                f.write(f"{len(nodes)} {' '.join(nodes)}\n")
            f.write("\n")

            f.write(f"CELL_TYPES {len(elements)}\n")
            for el in elements:
                el_type = get_element_type_from_numad(el)
                vtk_type = ELEMENTS_TO_VTK[el_type]
                f.write(f"{vtk_type}\n")
            f.write("\n")

        def _write_cells_data(f, cell_sets, total_cells):
            region_data = np.full(total_cells, -1, dtype=int)

            for region_value, (set_name, elements) in enumerate(cell_sets.items()):
                if "all" not in set_name:
                    region_data[elements] = region_value

            f.write(f"CELL_DATA {total_cells}\n")
            f.write("SCALARS CellRegions int 1\n")
            f.write("LOOKUP_TABLE default\n")

            for value in region_data:
                f.write(f"{value}\n")
            f.write("\n")

        def _write_nodes_data(f, node_sets, total_nodes):
            region_data = np.full(total_nodes, -1, dtype=int)

            sorted_sets = sorted(node_sets.items(), key=lambda item: len(item[1]), reverse=True)

            for region_value, (_, nodes) in enumerate(sorted_sets):
                region_data[nodes] = region_value

            f.write(f"POINT_DATA {total_nodes}\n")
            f.write("SCALARS NodeRegions int 1\n")
            f.write("LOOKUP_TABLE default\n")

            np.savetxt(f, region_data, fmt="%d")
            f.write("\n")

        with open(filename, "w") as f:
            f.write("# vtk DataFile Version 2.01\n")
            f.write("Unstructured Grid Example\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            _write_points(f, self.nodes)
            f.write("\n")
            _write_cells(f, self.elements)
            f.write("\n")
            _write_cells_data(f, self.element_sets, len(self.elements))
            f.write("\n")
            _write_nodes_data(f, self.node_sets, len(self.nodes))

    def plot(self, *args, **kwargs) -> pv.Plotter:
        self.shell_mesh()
        return plot_mesh(self)

    def _triangulate_mesh(self) -> None:
        """
        Converts all quadrilateral elements into triangles while preserving the element sets and mesh structure.

        Parameters:
        -----------
        mesh : PyNuMADMesh
            The mesh object containing the quadrilateral elements to be converted into triangles. This mesh should
            contain elements of type quadrilateral (4 nodes) which will be transformed into two triangles (3 nodes each).

        Returns:
        --------
        PyNuMADMesh
            A new PyNuMADMesh object where all quadrilateral elements are converted into 3-node triangle elements.
            The mesh structure is preserved, including the element sets and node associations.

        Notes:
        ------
        This function is useful when triangularization is required for compatibility with certain solvers or when
        simplifying the mesh. It ensures that:
        - The mesh's structural integrity is maintained during the conversion process.
        - Element sets (i.e., groups of elements) are updated to reflect the new triangular elements.
        - Node associations are correctly adjusted for the newly created triangular elements.
        """
        triangles = []
        el_id = 0
        elements_map = {}
        mesh = self.mesh.copy()
        with Progress() as progress:
            triangulate_task = progress.add_task(
                "Triangulating quad elements", total=len(mesh["elements"])
            )
            for i, element in enumerate(mesh["elements"]):
                if len(element) == 4 and element[3] != -1:
                    node0, node1, node2, node3 = element
                    first_triangle = [node0, node1, node2]
                    second_triangle = [node0, node2, node3]
                    elements_map[i] = ((el_id, first_triangle), (el_id + 1, second_triangle))
                    el_id += 2
                elif len(element) == 4 and element[3] == -1:
                    elements_map[i] = [(el_id, element[0:3])]
                    el_id += 1
                progress.update(triangulate_task, advance=1)

            elements = []
            update_elements_map_task = progress.add_task(
                "Updating elements map", total=len(elements_map.values())
            )
            for triangles in elements_map.values():
                for _, triangle in triangles:
                    elements.append([int(i) for i in triangle])
                progress.update(update_elements_map_task, advance=1)

            mesh["elements"] = np.array(elements)

            update_mesh_sets_task = progress.add_task(
                "Updating elements sets", total=len(mesh["sets"]["element"])
            )
            for elset in mesh["sets"]["element"]:
                new_labels = []
                for id in elset["labels"]:
                    el = elements_map[id]
                    new_labels.extend([l[0] for l in el])
                elset["labels"] = new_labels
                progress.update(update_mesh_sets_task, advance=1)

        self._mesh = mesh

    def _add_mid_nodes(self) -> None:
        """
        Converts shell elements (triangular) into quadratic elements by adding mid-edge nodes and ensures
        the correct orientation of the normal vectors.

        Parameters:
        -----------
        mesh : PyNuMADMesh
            The mesh object containing the shell elements (triangular) to be converted into quadratic elements.
            The mesh must also contain the nodes and elements that will be updated with new mid-edge nodes
            and adjusted normal orientations.

        Returns:
        --------
        PyNuMADMesh
            A new mesh object where 3-node triangular elements are converted into 6-node quadratic triangular
            elements. The normal vectors of the mesh are also adjusted to ensure correct orientation.

        Notes:
        ------
        This function is primarily used to convert linear triangular shell elements into quadratic ones.
        The process involves adding mid-edge nodes, which are necessary for the quadratic elements.
        Additionally, it ensures that the normal vectors of the elements are correctly oriented.

        The conversion is performed by calculating the midpoints of the edges of each triangular element,
        querying for existing nodes, and creating new nodes if necessary. If the distance between a calculated
        midpoint and an existing node is smaller than a threshold, the existing node is reused. Otherwise,
        a new node is added to the mesh.

        This function is crucial for simulations where higher-order elements are required, improving
        accuracy and convergence in structural and aeroelastic analyses.

        **Currently, only triangular elements are supported for this operation.**
        """
        mesh = self.mesh.copy()
        nodes = self.nodes.copy()
        elements = self.elements.copy()
        nodes_map = dict(enumerate(nodes)).copy()
        elements_map = dict(enumerate(elements)).copy()

        kdtree = KDTree(nodes)
        with Progress() as progress:
            task = progress.add_task("Converting Linear to Quadratic elements", total=len(elements))
            for e_id, element in elements_map.items():
                new_node_indices = []

                if e_id in self.elements_class[Elements.TRIANGLE]:
                    if len(element) == 3:
                        n1, n2, n3 = [int(i) for i in element]
                    else:
                        n1, n2, n3, _ = [int(i) for i in element]

                    mids = np.array(
                        [
                            (nodes_map[n1] + nodes_map[n2]) / 2,
                            (nodes_map[n2] + nodes_map[n3]) / 2,
                            (nodes_map[n3] + nodes_map[n1]) / 2,
                        ]
                    )
                elif e_id in self.elements_class[Elements.QUAD]:
                    n1, n2, n3, n4 = [int(i) for i in element]

                    mids = np.array(
                        [
                            (nodes_map[n1] + nodes_map[n2]) / 2,
                            (nodes_map[n2] + nodes_map[n3]) / 2,
                            (nodes_map[n3] + nodes_map[n4]) / 2,
                            (nodes_map[n4] + nodes_map[n1]) / 2,
                        ]
                    )
                else:
                    continue

                distance, mid_idxs = kdtree.query(mids)
                for i, mid in enumerate(mid_idxs):
                    if distance[i] < 1e-6:
                        new_node_indices.append(mid)
                    else:
                        new_node_index = len(nodes_map)
                        nodes_map[new_node_index] = mids[i]
                        new_node_indices.append(new_node_index)

                if distance.any() < 1e-6:
                    kdtree = KDTree(np.array(list(nodes_map.values())))

                if e_id in self.elements_class[Elements.TRIANGLE]:
                    elements_map[e_id] = np.array([n1, n2, n3, -1, *new_node_indices, -1])
                else:
                    elements_map[e_id] = np.array([n1, n2, n3, n4, *new_node_indices])

                progress.update(task, advance=1)

        mesh["nodes"] = np.array(list(nodes_map.values()))
        mesh["elements"] = np.array(list(elements_map.values()))
        self._mesh = mesh


class SquareShapeMesh(BaseMesh):
    """
    A class for generating quadrilateral/triangular meshes using Gmsh.

    Attributes
    ----------
    width : float
        Width of the domain.
    height : float
        Height of the domain.
    nx : int
        Number of elements along the x-direction.
    ny : int
        Number of elements along the y-direction.
    nodes : np.ndarray
        Array containing node coordinates.
    elements : np.ndarray
        Array containing element connectivity.
    dim : int
        Mesh dimension, dim = 2.
    """

    def __init__(
        self,
        width: float,
        height: float,
        nx: int,
        ny: int,
        quadratic: bool = True,
        triangular: bool = False,
    ):
        super().__init__(quadratic, triangular)
        self.dim = 2
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny

    def shell_mesh(self):
        """Generates the mesh using Gmsh with specified parameters"""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
            gmsh.model.add("rectangle")

            # Create geometry
            x0 = -self.width / 2
            x1 = self.width / 2
            y0 = 0.0
            y1 = self.height

            # Add points
            p1 = gmsh.model.geo.addPoint(x0, y0, 0)
            p2 = gmsh.model.geo.addPoint(x1, y0, 0)
            p3 = gmsh.model.geo.addPoint(x1, y1, 0)
            p4 = gmsh.model.geo.addPoint(x0, y1, 0)

            # Create boundary lines
            bottom = gmsh.model.geo.addLine(p1, p2)
            right = gmsh.model.geo.addLine(p2, p3)
            top = gmsh.model.geo.addLine(p3, p4)
            left = gmsh.model.geo.addLine(p4, p1)

            # Create surface
            loop = gmsh.model.geo.addCurveLoop([bottom, right, top, left])
            surface = gmsh.model.geo.addPlaneSurface([loop])

            # Set physical groups for boundaries
            gmsh.model.addPhysicalGroup(1, [top], name="top")
            gmsh.model.addPhysicalGroup(1, [bottom], name="bottom")
            gmsh.model.addPhysicalGroup(1, [left], name="left")
            gmsh.model.addPhysicalGroup(1, [right], name="right")
            gmsh.model.addPhysicalGroup(2, [surface], name="domain")

            # Configure mesh
            gmsh.model.geo.mesh.setTransfiniteCurve(bottom, self.nx + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(top, self.nx + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(left, self.ny + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(right, self.ny + 1)
            gmsh.model.geo.mesh.setTransfiniteSurface(surface, "Right", [p1, p2, p3, p4])

            if self._triangular_elements:
                gmsh.option.setNumber("Mesh.Algorithm", 6)
            else:
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)

            if self._quadratic_elements:
                gmsh.option.setNumber("Mesh.ElementOrder", 2)

            # Generate mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)

            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            nodes_array = np.array(coords, dtype=np.float64).reshape(-1, 3)
            self._mesh["nodes"] = nodes_array

            elem_types = gmsh.model.mesh.getElementTypes()
            elements = []
            for elem_type in elem_types:
                element_properties = gmsh.model.mesh.getElementProperties(elem_type)
                if element_properties[1] == 2:  # Verifica si la dimensión del elemento es 2
                    _, node_tags_elem = gmsh.model.mesh.getElementsByType(elem_type)
                    elements.append(node_tags_elem.reshape(-1, element_properties[3]))

            self._mesh["elements"] = np.array(elements[0], dtype=np.int32) - 1

            self._create_node_sets(gmsh)
        finally:
            gmsh.finalize()
        return self

    def _create_node_sets(self, gmsh):
        """Extracts node sets from Gmsh physical groups"""
        physical_groups = gmsh.model.getPhysicalGroups()
        node_sets = {}

        for dim, tag in physical_groups:
            if dim == 1:  # Boundary curves
                name = gmsh.model.getPhysicalName(dim, tag)
                nodes = []
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                for e in entities:
                    node_tags, _, _ = gmsh.model.mesh.getNodes(dim=1, tag=e)
                    nodes.extend([nt - 1 for nt in node_tags])
                node_sets[name] = nodes

        # Get all nodes in the mesh
        all_nodes = set(range(self._mesh["nodes"].shape[0]))

        # Get boundary nodes
        boundary_nodes = set()
        for boundary in ["top", "bottom", "left", "right"]:
            boundary_nodes.update(node_sets.get(boundary, []))

        # Surface nodes are all nodes except boundary nodes
        surface_nodes = list(all_nodes - boundary_nodes)

        # Store all sets
        self._mesh.setdefault("sets", {})["node"] = [
            {"name": "all", "labels": list(all_nodes)},
            {"name": "top", "labels": node_sets.get("top", [])},
            {"name": "bottom", "labels": node_sets.get("bottom", [])},
            {"name": "left", "labels": node_sets.get("left", [])},
            {"name": "right", "labels": node_sets.get("right", [])},
            {"name": "surface", "labels": surface_nodes},
        ]

    @classmethod
    def create_rectangle(
        cls,
        width: float,
        height: float,
        nx: int,
        ny: int,
        quadratic: bool = True,
        triangular: bool = False,
    ):
        return cls(width, height, nx, ny, quadratic, triangular)

    @classmethod
    def create_unit_square(
        cls,
        nx: int,
        ny: int,
        quadratic: bool = True,
        triangular: bool = False,
    ):
        return cls(1.0, 1.0, nx, ny, quadratic, triangular)
