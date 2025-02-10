from typing import Self

import numpy as np
import pynumad as pynu
from rich.progress import Progress
from scipy.spatial import KDTree

from turbine_mesher.enums import Elements
from turbine_mesher.mesh import BaseMesh
from turbine_mesher.types import PyNuMADBlade

__all__ = ["Blade"]


class Blade(BaseMesh):
    """
    A class representing the geometry of a blade for finite element analysis (FEA), typically used in aeroelastic
    analysis or structural simulations.

    Parameters:
    -----------
    yaml_file : str
        The path to the YAML configuration file that contains specifications for the blade geometry and mesh properties,
        formatted in windIO format. For more details on the format, please refer to:
        https://windio.readthedocs.io/en/stable/source/blade.html.
    use_quadratic_elements : bool, optional, default=True
        A flag indicating whether to use quadratic elements in the mesh. If True, quadratic elements are used. If False,
        linear elements are employed for the mesh.
    element_size : float, optional, default=0.5
        The size of the elements in the mesh, controlling the resolution of the mesh for the blade geometry. Smaller values
        result in finer meshes, while larger values lead to coarser meshes.
    n_samples : int, optional, default=300
        The number of samples to be used for discretization or analysis. This parameter can influence the resolution of
        the analysis or the density of the sampling.

    Attributes:
    -----------
    mesh : PyNuMADMesh
        A mesh object representing the blade geometry. The mesh consists of triangular elements, which can be either
        linear or quadratic depending on the `use_quadratic_elements` flag.
    blade : PyNuMADBlade
        The PyNuMAD Blade Object, representing the geometrical and mesh characteristics of the blade as defined by the
        provided configuration and mesh settings.
    """

    def __init__(
        self,
        yaml_file: str,
        element_size: float = 0.5,
        use_quadratic_elements: bool = True,
        enforce_triangular_elements: bool = False,
        n_samples: int = 300,
    ):
        """
        Initializes the Blade object by generating the blade geometry and mesh according to the provided parameters
        and the configuration specified in the YAML file.

        Parameters:
        -----------
        yaml_file : str
            The path to the YAML file that contains the blade geometry and mesh specifications. The file should follow
            the windIO format for blade geometry as described in the documentation:
            https://windio.readthedocs.io/en/stable/source/blade.html.

        use_quadratic_elements : bool, optional, default=True
            A flag to specify whether to use quadratic elements for the mesh. If True, quadratic elements are used;
            if False, linear elements are used.

        element_size : float, optional, default=0.5
            The desired size of the elements in the mesh. This value influences the mesh resolution, where smaller values
            result in a finer mesh and larger values lead to a coarser mesh.

        n_samples : int, optional, default=300
            The number of sample points to be used for discretizing the blade geometry. This parameter affects the density
            of the mesh and the resolution of the geometry analysis.

        Notes:
        ------
        This constructor processes the YAML file to extract the geometry and mesh settings, and then uses these settings
        to create the appropriate mesh for the blade geometry. The type of elements (linear or quadratic) is determined
        by the `use_quadratic_elements` flag.
        """
        super().__init__()

        self._yaml = yaml_file
        self._blade = pynu.Blade(yaml_file)
        self._blade.update_blade()
        self._qudratic_elements = use_quadratic_elements
        self._enforce_triangular_elements = enforce_triangular_elements

        self._mesh_element_size = element_size

        for stat in self._blade.definition.stations:
            stat.airfoil.resample(n_samples=n_samples)
        self._blade.update_blade()

        nStations = self._blade.geometry.coordinates.shape[2]
        minTELengths = 0.001 * np.ones(nStations)
        self._blade.expand_blade_geometry_te(minTELengths)

    @property
    def blade(self) -> PyNuMADBlade:
        """
        Retrieves the blade object representing the geometry of the blade.

        Returns:
        --------
        PyNuMADBlade
            The blade geometry object, which is a PyNuMADBlade instance that holds the blade's structural details.

        Notes:
        ------
        This property provides access to the blade geometry, which is typically created or loaded during the initialization
        of the Blade object from the YAML configuration file.
        """
        return self._blade

    def shell_mesh(self) -> Self:
        """
        Generates a shell mesh using PyNuMAD, based on the blade geometry and specified configuration.

        Returns:
        --------
        PyNuMADMesh
            A mesh object of type PyNuMADMesh, which represents the generated shell mesh for finite element analysis
            using shell elements.

        Notes:
        ------
        This method generates a shell mesh for the blade geometry, adjusting the mesh based on the configuration
        specified during initialization. The resulting mesh is used for structural or aeroelastic analysis.
        The following steps are performed:
        - A basic shell mesh is created using the PyNuMAD mesh generation function.
        - The mesh is then triangulated if necessary, using the `__triangulate` method to convert quadrilateral
          elements into triangles.
        - If quadratic elements are specified, mid-nodes are added to the mesh for higher-order accuracy using
          the `__add_mid_nodes` method.

        The mesh generated is stored in the `_mesh` attribute for subsequent use in the analysis.
        """
        adhes = 1

        self._mesh = pynu.mesh_gen.mesh_gen.get_shell_mesh(
            self._blade, adhes, self._mesh_element_size
        )
        if self._enforce_triangular_elements:
            self.__triangulate_mesh()

        if self._qudratic_elements:
            self.__add_mid_nodes()

        web_elements = [
            eset for eset in self._mesh["sets"]["element"] if "allShearWebEls" in eset["name"]
        ][0]["labels"]
        web_elements_nodes = [
            node for el in web_elements for node in self.elements[el] if node != -1
        ]
        surface_elements = [
            eset for eset in self._mesh["sets"]["element"] if "allOuterShellEls" in eset["name"]
        ][0]["labels"]
        surface_elements_nodes = [
            node for el in surface_elements for node in self.elements[el] if node != -1
        ]

        nset = self._mesh["sets"]["node"].pop()
        nset["labels"] = np.where(np.abs(self._mesh["nodes"][:, 2]) < 1e-3)[0]
        self._mesh["sets"]["node"].append(nset)

        self._mesh["sets"]["node"].append(
            {"name": "allShearWebNodes", "labels": web_elements_nodes}
        )

        self._mesh["sets"]["node"].append(
            {
                "name": "allOuterShellNodes",
                "labels": surface_elements_nodes,
            }
        )
        return self

    def __triangulate_mesh(self) -> None:
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

    def __add_mid_nodes(self) -> None:
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
