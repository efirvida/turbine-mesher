from enum import StrEnum

from pyvista.core.celltype import CellType


class ElementsTypes(StrEnum):
    TRIANGLE = "triangle"
    TRIANGLE6 = "triangle6"
    QUAD = "quad"
    QUAD8 = "quad8"
    QUAD9 = "quad9"


class CalculiXElementTypes(StrEnum):
    TRIANGLE = "S3"  # Triangle S3
    TRIANGLE6 = "S6"  # Triangle6 S6
    QUAD = "S4"  # Quad S4
    QUAD8 = "S8R"  # Quad8 S8


ELEMENTS_TO_CALCULIX = {
    ElementsTypes.TRIANGLE: CalculiXElementTypes.TRIANGLE,
    ElementsTypes.TRIANGLE6: CalculiXElementTypes.TRIANGLE6,
    ElementsTypes.QUAD: CalculiXElementTypes.QUAD,
    ElementsTypes.QUAD8: CalculiXElementTypes.QUAD8,
}

ELEMENTS_TO_VTK = {
    ElementsTypes.TRIANGLE: CellType.TRIANGLE,
    ElementsTypes.TRIANGLE6: CellType.QUADRATIC_TRIANGLE,
    ElementsTypes.QUAD: CellType.QUAD,
    ElementsTypes.QUAD8: CellType.QUADRATIC_QUAD,
    ElementsTypes.QUAD9: CellType.LAGRANGE_QUADRILATERAL,
}
