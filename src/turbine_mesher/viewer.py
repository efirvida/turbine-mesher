import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvista import themes
from pyvistaqt import QtInteractor

from .enums import ELEMENTS_TO_VTK
from .helpers import get_element_type_from_numad

__all__ = ["plot_mesh"]


class MeshViewer(QWidget):
    """A widget for visualizing finite element meshes with node and element sets.

    Attributes
    ----------
    mesh : Any
        The input mesh data containing nodes, elements, and sets.
    current_grid : pyvista.UnstructuredGrid or None
        The currently displayed grid in the plotter.
    is_2D_mesh : bool
        Flag indicating if the mesh is 2D (all Z coordinates are zero).
    plotter : pyvistaqt.QtInteractor
        The PyVista Qt interactor for 3D visualization.
    """

    def __init__(self, mesh):
        """Initialize the MeshViewer with given mesh data.

        Parameters
        ----------
        mesh : Any
            The mesh object containing nodes, elements, and sets information.
        """
        super().__init__()
        self.mesh = mesh
        self.current_grid = None
        self.is_2D_mesh = not bool(self.mesh.nodes[:, 2].sum())
        self.init_ui()
        self.create_set_table()
        self.update_plot()

    def init_ui(self):
        """Initialize the user interface components and layout."""
        self.setWindowTitle("Mesh Viewer")
        self.setMinimumSize(800, 600)

        # Main layout using QSplitter for resizable panels
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Control panel (left)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Visualization mode selector
        control_layout.addWidget(QLabel("Visualization Mode:"))
        self.vis_mode_combo = QComboBox()
        self.vis_mode_combo.addItems(["Surface with edges", "Wireframe", "Surface", "Points"])
        self.vis_mode_combo.setToolTip("Select mesh visualization style")
        self.vis_mode_combo.currentIndexChanged.connect(self.update_plot)
        control_layout.addWidget(self.vis_mode_combo)

        # Sets group box
        self.sets_group = QGroupBox("Sets (Nodes and Elements)")
        sets_layout = QVBoxLayout()

        # Set filter
        self.filter_le = QLineEdit()
        self.filter_le.setPlaceholderText("Filter sets by name or type")
        self.filter_le.textChanged.connect(self.filter_table)
        sets_layout.addWidget(self.filter_le)

        # Sets table
        self.sets_table = QTableWidget()
        self.sets_table.setColumnCount(3)
        self.sets_table.setHorizontalHeaderLabels(["Set", "Type"])
        self.sets_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.sets_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.sets_table.itemChanged.connect(self.update_plot)
        sets_layout.addWidget(self.sets_table)

        self.sets_group.setLayout(sets_layout)
        control_layout.addWidget(self.sets_group)

        splitter.addWidget(control_widget)

        # PyVista visualization panel (right)
        pv.set_plot_theme(themes.ParaViewTheme())
        self.plotter = QtInteractor(parent=self)
        splitter.addWidget(self.plotter)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_set_table(self):
        """Populate the table with node and element sets from the mesh."""
        node_sets = list(getattr(self.mesh, "node_sets", {}).keys())
        element_sets = list(getattr(self.mesh, "element_sets", {}).keys())
        total_rows = len(node_sets) + len(element_sets)

        self.sets_table.blockSignals(True)
        self.sets_table.setRowCount(total_rows)
        row = 0
        # Add node sets
        for set_name in node_sets:
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            chk_item.setText(set_name)
            type_item = QTableWidgetItem("Node")
            type_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            self.sets_table.setItem(row, 0, chk_item)
            self.sets_table.setItem(row, 1, type_item)
            row += 1
        # Add element sets
        for set_name in element_sets:
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Unchecked)
            chk_item.setText(set_name)
            type_item = QTableWidgetItem("Element")
            type_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            self.sets_table.setItem(row, 0, chk_item)
            self.sets_table.setItem(row, 1, type_item)
            row += 1

        self.sets_table.blockSignals(False)

    def filter_table(self, text: str):
        """Filter the sets table based on input text.

        Parameters
        ----------
        text : str
            The filter text to match against set names and types.
        """
        text = text.lower()
        for row in range(self.sets_table.rowCount()):
            name = self.sets_table.item(row, 0).text().lower()
            set_type = self.sets_table.item(row, 1).text().lower()
            self.sets_table.setRowHidden(row, text not in name and text not in set_type)

    def get_selected_sets(self) -> tuple[list[str], list[str]]:
        """Get currently selected sets from the table.

        Returns
        -------
        tuple[list[str], list[str]]
            Two lists containing selected node set names and element set names.
        """
        node_sets, element_sets = [], []
        for row in range(self.sets_table.rowCount()):
            if (
                not self.sets_table.isRowHidden(row)
                and self.sets_table.item(row, 0).checkState() == Qt.Checked
            ):
                set_name = self.sets_table.item(row, 0).text()
                if self.sets_table.item(row, 1).text() == "Node":
                    node_sets.append(set_name)
                else:
                    element_sets.append(set_name)
        return node_sets, element_sets

    def _create_element_grid(self, element_ids: set[int] = None) -> pv.UnstructuredGrid | None:
        """Create PyVista grid from specified element IDs.

        Parameters
        ----------
        element_ids : set of int, optional
            IDs of elements to include in the grid. If None, includes all elements.

        Returns
        -------
        pyvista.UnstructuredGrid or None
            Constructed unstructured grid, or None if no valid elements found.
        """
        cells, cell_types = [], []
        for el_type, ids in getattr(self.mesh, "elements_class", {}).items():
            for el_id in ids:
                if element_ids is not None and el_id not in element_ids:
                    continue
                if el_id >= len(self.mesh.elements):
                    continue

                element = [n for n in self.mesh.elements[el_id] if n != -1]
                if element:
                    cells.append([len(element)] + element)
                    el_type = get_element_type_from_numad(element)
                    cell_types.append(ELEMENTS_TO_VTK[el_type])

        return (
            pv.UnstructuredGrid(
                np.hstack(cells).astype(np.int32),
                np.array(cell_types, dtype=np.uint8),
                np.array(self.mesh.nodes, dtype=np.float64),
            )
            if cells
            else None
        )

    def _create_node_points(self, node_ids: set[int]) -> pv.PolyData | None:
        """Create point cloud from specified node IDs.

        Parameters
        ----------
        node_ids : set of int
            IDs of nodes to include in the point cloud.

        Returns
        -------
        pyvista.PolyData or None
            Points data or None if no valid nodes found.
        """
        if not node_ids:
            return None

        nodes = np.array(self.mesh.nodes, dtype=np.float64)
        if nodes.shape[1] != 3:
            nodes = np.concatenate((nodes, np.zeros((nodes.shape[0], 1))), axis=1)

        valid_ids = [i for i in node_ids if i < len(nodes)]
        return pv.PolyData(nodes[valid_ids]) if valid_ids else None

    def _add_custom_axes(self, label_size=20):
        """Add customized axes to the plotter.

        Parameters
        ----------
        label_size : int, optional
            Font size for axis labels, by default 20
        """
        self.plotter.show_axes()

    @Slot()
    def update_plot(self):
        """Update the 3D visualization based on current selections."""
        try:
            self.plotter.clear()

            # Set visualization style
            vis_mode = self.vis_mode_combo.currentText()
            style = {}
            if vis_mode == "Wireframe":
                style = {"style": "wireframe"}
            elif vis_mode == "Points":
                style = {"style": "points", "render_points_as_spheres": True, "point_size": 10}
            elif "Surface" in vis_mode:
                style = {"style": "surface"}
                if "edges" in vis_mode:
                    style.update({"show_edges": True, "edge_color": "black"})

            # Get selected sets
            node_sets, element_sets = self.get_selected_sets()

            # Process elements
            if element_sets:
                element_ids = set()
                for s in element_sets:
                    element_ids.update(self.mesh.element_sets.get(s, []))

                if grid := self._create_element_grid(element_ids):
                    self.plotter.add_mesh(grid, name="elements", **style)

            # Process nodes
            if node_sets:
                node_ids = set()
                for s in node_sets:
                    node_ids.update(self.mesh.node_sets.get(s, []))

                if points := self._create_node_points(node_ids):
                    self.plotter.add_points(
                        points, name="nodes", **{"style": "points", "point_size": 10}
                    )

            # Show full mesh if no selections
            if not element_sets and not node_sets:
                if grid := self._create_element_grid():
                    self.plotter.add_mesh(grid, name="surface", **style)

            # Set view orientation
            self.plotter.view_xy() if self.is_2D_mesh else self.plotter.view_xz()

            # Finalize plot
            light = pv.Light(position=(0, 0, 1), light_type="camera light")
            self._add_custom_axes()
            self.plotter.add_light(light)
            self.plotter.reset_camera()
            self.plotter.render()

        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Failed to update plot:\n{str(e)}")


def plot_mesh(mesh):
    """Launch the mesh visualization application.

    Parameters
    ----------
    mesh : Any
        The mesh object to visualize.

    Returns
    -------
    None
    """
    app = QApplication.instance() or QApplication()
    viewer = MeshViewer(mesh)
    viewer.show()
    app.exec()
    return None
