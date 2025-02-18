from typing import Self

import numpy as np
import pynumad as pynu

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
        quadratic: bool = True,
        triangular: bool = False,
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
        super().__init__(quadratic, triangular)

        self._yaml = yaml_file
        self._blade = pynu.Blade(yaml_file)
        self._blade.update_blade()

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
        if self._triangular_elements:
            self._triangulate_mesh()

        if self._quadratic_elements:
            self._add_mid_nodes()

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
