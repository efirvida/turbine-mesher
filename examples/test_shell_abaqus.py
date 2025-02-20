from abaqus import *
from abaqusConstants import *
from caeModules import *

# Parámetros del problema
WIDTH = 1.0
HEIGHT = 1.0
CELLS_PER_UNIT = 50
NX, NY = int(WIDTH * CELLS_PER_UNIT), int(HEIGHT * CELLS_PER_UNIT)
E = 210e3  # Módulo de Young en MPa
NU = 0.3  # Coeficiente de Poisson
LOAD = 50  # Carga en dirección Z en N/m^2

# Crear modelo
model_name = "MITC4_Model"
mdb.Model(name=model_name)
model = mdb.models[model_name]

# Crear la pieza (Placa 2D)
part_name = "Plate"
model.ConstrainedSketch(name="__profile__", sheetSize=10.0)
model.sketches["__profile__"].rectangle(point1=(0.0, 0.0), point2=(WIDTH, HEIGHT))
model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
model.parts[part_name].BaseShell(sketch=model.sketches["__profile__"])
del model.sketches["__profile__"]

# Crear material y sección
material_name = "Steel"
model.Material(name=material_name)
model.materials[material_name].Elastic(table=[(E, NU)])
model.HomogeneousShellSection(
    name="PlateSection",
    preIntegrate=OFF,
    material=material_name,
    thickness=0.01,
    thicknessType=UNIFORM,
)

# Asignar sección a la pieza
part = model.parts[part_name]
face = part.faces.findAt(((WIDTH / 2, HEIGHT / 2, 0.0),))
region = part.Set(faces=face, name="PlateRegion")
part.SectionAssignment(region=region, sectionName="PlateSection")

# Mallado con MITC4 (S4)
part.seedPart(size=WIDTH / NX, deviationFactor=0.1, minSizeFactor=0.1)
part.setElementType(
    regions=region, elemTypes=(ElemType(elemCode=S4, elemLibrary=STANDARD),)
)
part.generateMesh()

# Crear ensamblaje
assembly = model.rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)
instance_name = "PlateInstance"
assembly.Instance(name=instance_name, part=part, dependent=ON)

# Condiciones de frontera
edges = assembly.instances[instance_name].edges
left_edge = edges.findAt(((0.0, HEIGHT / 2, 0.0),))
right_edge = edges.findAt(((WIDTH, HEIGHT / 2, 0.0),))
top_edge = edges.findAt(((WIDTH / 2, HEIGHT, 0.0),))
bottom_edge = edges.findAt(((WIDTH / 2, 0.0, 0.0),))

assembly.Set(edges=left_edge, name="LeftEdge")
assembly.Set(edges=right_edge, name="RightEdge")
assembly.Set(edges=top_edge, name="TopEdge")
assembly.Set(edges=bottom_edge, name="BottomEdge")

model.DisplacementBC(
    name="BC_Left",
    createStepName="Initial",
    region=assembly.sets["LeftEdge"],
    u1=0.0,
    u2=0.0,
    u3=0.0,
    ur1=0.0,
    ur2=0.0,
    ur3=0.0,
)

model.DisplacementBC(
    name="BC_Right",
    createStepName="Initial",
    region=assembly.sets["RightEdge"],
    u1=0.0,
    u2=0.0,
    u3=0.0,
    ur1=0.0,
    ur2=0.0,
    ur3=0.0,
)

model.DisplacementBC(
    name="BC_Top",
    createStepName="Initial",
    region=assembly.sets["TopEdge"],
    u1=0.0,
    u2=0.0,
    u3=0.0,
    ur1=0.0,
    ur2=0.0,
    ur3=0.0,
)

model.DisplacementBC(
    name="BC_Bottom",
    createStepName="Initial",
    region=assembly.sets["BottomEdge"],
    u1=0.0,
    u2=0.0,
    u3=0.0,
    ur1=0.0,
    ur2=0.0,
    ur3=0.0,
)

# Carga de presión en la cara superior
model.StaticStep(name="LoadStep", previous="Initial")
face = assembly.instances[instance_name].faces.findAt(((WIDTH / 2, HEIGHT / 2, 0.0),))
region = assembly.Surface(side1Faces=face, name="LoadSurface")
model.Pressure(
    name="PressureLoad", createStepName="LoadStep", region=region, magnitude=LOAD
)

# Configuración y ejecución del análisis
job_name = "MITC4_Analysis"
mdb.Job(
    name=job_name,
    model=model_name,
    type=ANALYSIS,
    explicitPrecision=SINGLE,
    nodalOutputPrecision=SINGLE,
    description="MITC4 Shell Element Analysis",
)
mdb.jobs[job_name].submit(consistencyChecking=OFF)
mdb.jobs[job_name].waitForCompletion()

# Post-procesamiento (Desplazamientos en Z)
odb = session.openOdb(name=job_name + ".odb")
step = odb.steps["LoadStep"]
frame = step.frames[-1]
displacement = frame.fieldOutputs["U"]
region = odb.rootAssembly.instances[instance_name.upper()].nodeSets[" ALL NODES"]
displacement_set = displacement.getSubset(region=region)
displacements = [node.data[2] for node in displacement_set.values]

# Graficar desplazamientos en Z
plt.figure()
plt.plot(displacements, label="Desplazamientos en Z")
plt.xlabel("Nodos")
plt.ylabel("Desplazamiento (m)")
plt.title("Desplazamientos en Z para MITC4")
plt.legend()
plt.grid(True)
plt.show()
