from enum import StrEnum

from pyvista.core.celltype import CellType


class Elements(StrEnum):
    TRIANGLE = "triangle"
    TRIANGLE6 = "triangle6"
    QUAD = "quad"
    QUAD8 = "quad8"


class CalculiXElementTypes(StrEnum):
    TRIANGLE = "S3"  # Triangle S3
    TRIANGLE6 = "S6"  # Triangle6 S6
    QUAD = "S4"  # Quad S4
    QUAD8 = "S8R"  # Quad8 S8


ELEMENTS_TO_CALCULIX = {
    Elements.TRIANGLE: CalculiXElementTypes.TRIANGLE,
    Elements.TRIANGLE6: CalculiXElementTypes.TRIANGLE6,
    Elements.QUAD: CalculiXElementTypes.QUAD,
    Elements.QUAD8: CalculiXElementTypes.QUAD8,
}

ELEMENTS_TO_VTK = {
    Elements.TRIANGLE: CellType.TRIANGLE,
    Elements.TRIANGLE6: CellType.QUADRATIC_TRIANGLE,
    Elements.QUAD: CellType.QUAD,
    Elements.QUAD8: CellType.QUADRATIC_QUAD,
}
