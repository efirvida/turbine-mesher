import os
from pathlib import Path

from turbine_mesher import Blade, Rotor

if __name__ == "__main__":
    from glob import glob

    CURRENT_FOLDER = Path().resolve()
    DATA_FOLDER = os.path.join(CURRENT_FOLDER, "data")
    OUTPUT_FOLDER = os.path.join(CURRENT_FOLDER, "output")

    yamls = glob(f"{DATA_FOLDER}/*.yaml")

    for blade_yaml_file in yamls:
        blade_name = os.path.basename(blade_yaml_file)[:-5]

        OUTPUT_INP = os.path.join(CURRENT_FOLDER, "output", f"Rotor_{blade_name}.inp")
        OUTPUT_VTK = os.path.join(CURRENT_FOLDER, "output", f"Rotor_{blade_name}.vtk")
        rotor = Rotor(blade_yaml_file, element_size=0.5)
        rotor.plot()

        rotor.write_mesh(OUTPUT_INP)
        rotor.write_mesh(OUTPUT_VTK)

        OUTPUT_INP = os.path.join(CURRENT_FOLDER, "output", f"{blade_name}.inp")
        OUTPUT_VTK = os.path.join(CURRENT_FOLDER, "output", f"{blade_name}.vtk")
        blade = Blade(blade_yaml_file, element_size=0.5)
        blade.write_mesh(OUTPUT_INP)
        blade.write_mesh(OUTPUT_VTK)
