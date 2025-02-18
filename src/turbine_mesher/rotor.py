from copy import deepcopy
from typing import Self

import numpy as np
import yaml

from turbine_mesher.blade import Blade
from turbine_mesher.helpers import rotate_around_y
from turbine_mesher.mesh import BaseMesh

__all__ = ["Rotor"]


class Rotor(BaseMesh):
    def __init__(
        self,
        yaml_file: str,
        element_size: float = 0.5,
        quadratic: bool = True,
        triangular: bool = False,
        n_samples: int = 300,
    ):
        super().__init__(quadratic, triangular)
        self._yaml = yaml_file
        self._blade = Blade(
            yaml_file,
            element_size=element_size,
            quadratic=quadratic,
            triangular=triangular,
            n_samples=n_samples,
        )
        self._blades = []

        with open(self._yaml) as blade_yaml:
            self.__turbine_definition = yaml.load(blade_yaml, Loader=yaml.Loader)

    @property
    def hub_diameter(self):
        try:
            return self.__turbine_definition["components"]["hub"]["diameter"]
        except KeyError:
            raise ValueError("Hub diameter ots not defined in YAML input file")

    @property
    def number_of_blades(self):
        try:
            return self.__turbine_definition["assembly"]["number_of_blades"]
        except KeyError:
            # set default number of blades to 3 if its not defined in the YAML input file
            return 3

    def shell_mesh(self) -> Self:
        base_blade = deepcopy(self._blade)
        _ = base_blade.mesh
        base_blade._mesh["nodes"][:, 2] += self.hub_diameter / 2
        nodes_count = len(base_blade.nodes)
        elements_count = len(base_blade.elements)

        for i in range(self.number_of_blades):
            angle = 2 * np.pi * i / self.number_of_blades
            nodes = rotate_around_y(base_blade.nodes.copy(), angle)
            new_blade = deepcopy(base_blade)
            new_blade._mesh["nodes"] = nodes
            self._blades.append(new_blade)

        total_nodes = base_blade.nodes.shape[0] * self.number_of_blades
        total_elements = base_blade.elements.shape[0] * self.number_of_blades
        self._mesh = {
            "nodes": np.zeros((total_nodes, 3), dtype=np.float64),
            "elements": np.zeros((total_elements, base_blade.elements.shape[1]), dtype=np.int32),
            "sets": deepcopy(base_blade.mesh["sets"]),
            "materials": base_blade.mesh["materials"],
            "sections": base_blade.mesh["sections"],
        }

        for i, blade in enumerate(self._blades):
            node_offset = i * nodes_count
            element_offset = i * elements_count

            start_node = node_offset
            end_node = node_offset + nodes_count
            self._mesh["nodes"][start_node:end_node] = blade.nodes

            blade_elements = blade.elements.copy()
            valid_mask = blade_elements != -1

            blade_elements[valid_mask] += node_offset

            start_element = element_offset
            end_element = element_offset + elements_count
            self._mesh["elements"][start_element:end_element] = blade_elements

        extended_node_set = []
        for node_set in self._mesh["sets"]["node"]:
            labels = np.array(node_set["labels"])
            for i in range(self.number_of_blades):
                extended_node_set.append(
                    {
                        "name": f"{node_set['name']}_Blade{i + 1}",
                        "labels": (labels + i * nodes_count).tolist(),
                    }
                )
        self._mesh["sets"]["node"] = extended_node_set

        extended_element_set = []
        for el_set in self._mesh["sets"]["element"]:
            labels = np.array(el_set["labels"])
            for i in range(self.number_of_blades):
                extended_element_set.append(
                    {
                        "name": f"{el_set['name']}_Blade{i + 1}",
                        "labels": (labels + i * elements_count).tolist(),
                    }
                )
        self._mesh["sets"]["element"] = extended_element_set

        new_sections = []
        for section in self._mesh["sections"]:
            el_set = section["elementSet"]
            for i in range(self.number_of_blades):
                new_sction = deepcopy(section)
                new_sction["elementSet"] = f"{el_set}_Blade{i + 1}"
                new_sections.append(new_sction)
        self._mesh["sections"] = new_sections

        return self
