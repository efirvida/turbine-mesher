import os
from typing import Union, Dict
import turbine_mesher.mesh as mesh
from ..mesh import pynumad_to_meshio, remesh_with_trimesh
import trimesh
import meshio
from ..types import MeshIOMesh, PyNuMADMesh


def write_mesh(mesh_obj: Union[PyNuMADMesh, MeshIOMesh], file_name: str):
    """
    Writes a solid mesh to a file, creating directories if needed.

    :param mesh_obj: Mesh object (MeshIO or PyNuMAD).
    :param file_name: File name to write the mesh.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Convert to meshio.Mesh if input is a PyNuMAD dictionary
    if isinstance(mesh_obj, dict):
        mesh = pynumad_to_meshio(mesh_obj)
    else:
        mesh = mesh_obj

    # Determine if remeshing is needed based on file extension
    file_extension = file_name.lower().split(".")[-1]
    if file_extension in {"stl", "obj"}:
        mesh = remesh_with_trimesh(mesh)

    # Write using meshio for other formats
    meshio.write(file_name, mesh)
