import pyvista as pv
import numpy as np
import meshio
import json
from pathlib import Path

# domain ids
porous_id = 1
fluid_id = 2

# facet ids
skull_id = 1
spinal_canal_id = 2
interface_id = 3
spinal_cord_id = 4
aqueduct_V4_id = 5

skull = pv.Sphere(radius = 0.08)
parenchyma = pv.Sphere(radius = 0.07)
ventricle = pv.Sphere(radius = 0.02)
canal = pv.Cylinder(center=(0, 0, -0.105), direction=(0,0,-1),
                        radius=0.025, height=0.08).triangulate()
cord = pv.Cylinder(center=(0, 0, -0.105), direction=(0,0,-1),
                        radius=0.017, height=0.08).triangulate()
aqueduct = pv.Cylinder(center=(0, 0.03, -0.03), direction=(0,1,-1),
                       radius=0.004, height=0.06).triangulate()

# save the surface meshes for later use by fTetWild
stl_directory = Path("mesh/stls/")
stl_directory.mkdir(exist_ok=True, parents=True)
parenchyma.save(stl_directory / "parenchyma.stl")
skull.save(stl_directory / "skull.stl")
cord.save(stl_directory / "cord.stl")
canal.save(stl_directory / "canal.stl")
ventricle.save(stl_directory / "ventricle.stl")
aqueduct.save(stl_directory / "aqueduct.stl")
print(f"STL files saved to {stl_directory.resolve()}")