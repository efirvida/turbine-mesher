import os

from turbine_mesher.mesh import mesh_rotor, shell_mesh, mesh_statistics, solid_mesh
from turbine_mesher.io.mesh import write_mesh
from pynumad.mesh_gen.mesh_tools import *
from turbine_mesher.mesh import Mesh


# ## Define inputs
CURRENT_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(CURRENT_FOLDER, "data")
OUTPUT_FOLDER = os.path.join(CURRENT_FOLDER, "output")
YAML_NAME = "IEA-15-240-RWT"


bladeYaml = os.path.join(DATA_FOLDER, f"{YAML_NAME}.yaml")
# solidMeshFile = os.path.join(OUTPUT_FOLDER, "mesh.vtk")
# shellMeshFile = os.path.join(OUTPUT_FOLDER, "shell_mesh.vtk")
# rotorSolidMeshFile = os.path.join(OUTPUT_FOLDER, "rotor_mesh.vtk")

# ## Read blade data from yaml file
# blade = pynu.Blade()
# blade.read_yaml(bladeYaml)

# hub = read_hub_data(bladeYaml)

# ## Set the airfoil point resolution
# for stat in blade.definition.stations:
#     stat.airfoil.resample(n_samples=300)

# # blade.generate_geometry()
# blade.update_blade()
# nStations = blade.geometry.coordinates.shape[2]
# minTELengths = 0.001 * np.ones(nStations)
# blade.expand_blade_geometry_te(minTELengths)

# ## Set the target element size for the mesh
# elementSize = 0.5
# blade_mesh = solid_mesh(blade, elementSize)
# # rotor = mesh_rotor(blade_mesh, 3, hub.radius)

# write_mesh(blade_mesh, solidMeshFile)
# # write_mesh(rotor, rotorSolidMeshFile)

# # mesh_statistics(rotor)

# shellMesh = shell_mesh(blade, elementSize)
# write_mesh(shellMesh, shellMeshFile)


##########################
# VISUALIZE WITH PyVISTA #
##########################

# pv.set_plot_theme("dark")
# mesh = pv.read(rotorSolidMeshFile)


# # Crear un objeto Plotter con el tema oscuro
# cplot = pv.Plotter(off_screen=False)
# cplot.set_background("black")  # Fondo negro para tema oscuro

# # Agregar la malla a la visualización, mostrando los bordes y color
# cplot.add_mesh(solidMeshFile, show_edges=True, color="lightblue")

# # Mostrar los ejes
# cplot.show_axes()

# # Mostrar la visualización
# cplot.show()

# import alphashape
# import matplotlib.pyplot as plt

# points = blade_mesh.points
# # Define alpha parameter
# alpha = 0.5

# # Generate the alpha shape
# alpha_shape = alphashape.alphashape(points, alpha)

# # Initialize plot
# fig, ax = plt.subplots()

# # Plot input points
# ax.scatter(points)

# plt.show()


mesh = Mesh(bladeYaml, 0.1)
mesh.shell_mesh()
# mesh.write(f"{OUTPUT_FOLDER}/shell.vtk")
mesh.show_statistics()
# mesh.solid_mesh()
# mesh.to_gmsh()
# mesh.write("output/solid.vtk")
# print(mesh)
# mesh.mesh_rotor(3)
# mesh.write("output/rotor.vtk")
# print(mesh)
