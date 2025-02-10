import os
from typing import Dict, List, Self, Set

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from .enums import ELEMENTS_TO_CALCULIX, ELEMENTS_TO_VTK, Elements
from .helpers import get_element_type_from_numad
from .types import PyNuMADMesh


class BaseMesh:
    def __init__(self) -> None:
        self._mesh = {}

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
        if not self._mesh:
            self.shell_mesh()
        return self._mesh

    @property
    def nodes(self) -> np.ndarray:
        return self.mesh["nodes"]

    @property
    def elements(self) -> np.ndarray:
        return np.array(self.mesh["elements"], dtype=np.int32)

    @property
    def element_sets(self) -> Dict[str, List]:
        return {el_set["name"]: el_set["labels"] for el_set in self.mesh["sets"]["element"]}

    @property
    def node_sets(self) -> Dict[str, List]:
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

        mesh = self.mesh
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

            for mat in mesh["materials"]:
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
                for sec in mesh["sections"]:
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

            for region_value, (set_name, nodes) in enumerate(sorted_sets):
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

    def plot(
        self, show=True, show_sets: bool = False, show_edges: bool = True, **kwargs
    ) -> pv.Plotter:
        """
        Visualiza la malla usando PyVista.

        Parámetros:
        -----------
        show_sets : bool, opcional
            Muestra los conjuntos de nodos y elementos como datos escalares.
        show_edges : bool, opcional
            Muestra las aristas de los elementos.
        **kwargs : dict
            Argumentos adicionales para pyvista.Plotter.

        Retorna:
        --------
        pyvista.Plotter
            Objeto plotter de PyVista con la malla cargada.
        """
        if not hasattr(self, "nodes") or not hasattr(self, "elements_class"):
            raise ValueError("La malla no contiene nodos o elementos válidos.")

        try:
            if len(self.nodes) == 0:
                raise ValueError("La malla no contiene nodos.")

            cells = []
            cell_types = []

            for el_type, el_ids in self.elements_class.items():
                if not el_ids:
                    continue

                valid_elements = [
                    [n for n in self.elements[el_id] if n != -1]
                    for el_id in el_ids
                    if el_id in self.elements
                ]

                for element in valid_elements:
                    cells.append([len(element)] + element)
                    cell_types.append(ELEMENTS_TO_VTK.get(el_type, None))

            if cells and all(t is not None for t in cell_types):
                cells = np.hstack(cells).astype(np.int64)
                cell_types = np.array(cell_types, dtype=np.uint8)
            else:
                raise ValueError("No se encontraron elementos válidos para visualizar.")

            points = np.array(self.nodes, dtype=np.float64)
            grid = pv.UnstructuredGrid(cells, cell_types, points)

            if show_sets:
                node_regions = np.full(len(self.nodes), -1, dtype=int)
                for i, nodes in enumerate(self.node_sets.values()):
                    node_regions[nodes] = i
                grid.point_data["Node Sets"] = node_regions

                cell_regions = np.full(len(cell_types), -1, dtype=int)
                for i, elements in enumerate(self.element_sets.values()):
                    cell_regions[elements] = i
                grid.cell_data["Element Sets"] = cell_regions

            plotter = pv.Plotter(**kwargs)
            plotter.add_mesh(
                grid,
                show_edges=show_edges,
                edge_color="black",
                color="white",
                opacity=0.8,
                scalars="Node Sets" if show_sets else None,
                render_lines_as_tubes=False,
            )

            if show_sets:
                plotter.add_scalar_bar(
                    title="Node Sets",
                    n_labels=len(self.node_sets),
                    italic=False,
                    bold=True,
                    title_font_size=20,
                    label_font_size=16,
                    color="black",
                )

            if show:
                plotter.show()

            return plotter

        except ImportError:
            raise ImportError("PyVista no está instalado. Instálalo con: pip install pyvista")
        except Exception as e:
            raise RuntimeError(f"Error al generar la visualización: {e}")


class SquareShapeMesh:
    """
    A class for generating and visualizing structured quadrilateral meshes.

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

    def __init__(self, width: float, height: float, nx: int, ny: int):
        """
        Initializes the mesh generator with domain dimensions and discretization parameters.

        Parameters
        ----------
        width : float
            Width of the domain.
        height : float
            Height of the domain.
        nx : int
            Number of elements along the x-direction.
        ny : int
            Number of elements along the y-direction.
        """
        self.dim = 2  # 2D mesh
        self.width = width
        self.height = height
        self.nx = nx
        self.ny = ny
        self.nodes = None
        self.elements = None
        self.generate_mesh()

    def generate_mesh(self):
        """
        Generates the quadrilateral mesh.
        """
        x = np.linspace(0, self.width, self.nx + 1)
        y = np.linspace(0, self.height, self.ny + 1)
        X, Y = np.meshgrid(x, y)
        self.nodes = np.column_stack([X.ravel(), Y.ravel()])

        elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                n1 = j * (self.nx + 1) + i
                n2 = n1 + 1
                n3 = n1 + (self.nx + 1)
                n4 = n3 + 1
                elements.append([n1, n2, n4, n3])

        self.elements = np.array(elements, dtype=int)

    def plot(self):
        """
        Plots the generated mesh.
        """
        fig, ax = plt.subplots()
        for element in self.elements:
            polygon = self.nodes[element]
            polygon = np.vstack([polygon, polygon[0]])  # Close the quadrilateral
            ax.plot(polygon[:, 0], polygon[:, 1], "k")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Quadrilateral Mesh")
        ax.set_aspect("equal")
        plt.show()

    @property
    def num_nodes(self):
        return self.nodes.shape[0]

    @property
    def x(self):
        return self.nodes[:, 0]

    @property
    def y(self):
        return self.nodes[:, 1]

    @classmethod
    def create_rectangle(cls, width: float, height: float, nx: int, ny: int):
        """
        Creates a quadrilateral mesh for a rectangular domain.

        Parameters
        ----------
        width : float
            Width of the rectangle.
        height : float
            Height of the rectangle.
        nx : int
            Number of elements along the x-axis.
        ny : int
            Number of elements along the y-axis.

        Returns
        -------
        MeshGenerator
            An instance of MeshGenerator with the specified parameters.
        """
        return cls(width, height, nx, ny)

    @classmethod
    def create_unit_square(cls, nx: int, ny: int):
        """
        Creates a quadrilateral mesh for a unit square domain (1x1).

        Parameters
        ----------
        nx : int
            Number of elements along the x-axis.
        ny : int
            Number of elements along the y-axis.

        Returns
        -------
        MeshGenerator
            An instance of MeshGenerator with the specified parameters.
        """
        return cls(1.0, 1.0, nx, ny)


# Example usage
if __name__ == "__main__":
    mesh = SquareShapeMesh.create_rectangle(width=2.0, height=1.0, nx=4, ny=2)
    mesh.plot()

    unit_mesh = SquareShapeMesh.create_unit_square(nx=5, ny=5)
    unit_mesh.plot()
