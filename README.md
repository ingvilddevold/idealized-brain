# Idealized Brain Meshing

This repository contains command-line tools for generating, tagging, and iteratively refining idealized multi-domain 3D brain meshes for Biot-Stokes-type simulations in FEniCSx.
The geometry corresponds to the one used in Chapter 4 of [*Mathematical Modelling of the Human Brain II: From Glymphatics to Deep Learning*](https://doi.org/10.1007/978-3-032-00679-0).

## Requirements
To install the required dependencies in a Conda environment, run
```bash
conda env create -f environment.yml
```
Then, activate the environment with
```bash
conda activate fenicsxmesh
```

## Tools
### 1. Mesh Generation (generateMesh.py)
Generates surface STLs using Pyvista and creates a volumetric FEniCSx mesh from these using CSG (Constructive Solid Geometry), also generating properly tagged subdomains and boundaries.

**Generate surface STLs:**

```bash
python generateMesh.py surfaces --output-dir surfaces
```
(Add `--show` to preview the geometry in a PyVista plot).

**Generate and tag the FEniCSx mesh:**

```bash
python generateMesh.py mesh --stl-dir surfaces --output-dir mesh
```
(Add `--separate-interfaces` to tag the pial membrane and ventricular walls (ependyma) with distinct IDs instead of a unified interface tag).

Note: It is also possible to generate the mesh from the surface STLs with fTetWild directly from command line as
```bash
ftetwild --csg csg.json --output mesh/idealizedBrainMesh -e 0.002 -l 0.05
```

### 2. Uniform Mesh Refinement (refineMesh.py)
Iteratively refines an existing FEniCSx mesh while correctly transferring cell (subdomain) and facet (boundary) tags to the newly created child meshes.

**Refine a mesh:**

```bash
python refineMesh.py --input-dir mesh --levels 2
```
This will read the base mesh and sequentially generate `Ref1`, `Ref2`, etc., in sibling directories.

### Output Structure
After generating and refining the mesh, the directory structure will look similar to this:
```plaintext
.
├── mesh/
│   ├── mesh.xdmf
│   └── mesh_boundaries.xdmf
├── meshRef1/
│   ├── meshRef1.xdmf
│   └── meshRef1_boundaries.xdmf
```

### FEniCSx Marker Reference

When loading the `.xdmf` files into FEniCSx, use the following integer IDs:

#### Subdomains (Cells)
| ID | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| **`1`** | Subdomain | Porous | Parenchyma |
| **`2`** | Subdomain | Fluid | Subarachnoid space and ventricles 


#### Boundaries and interfaces (Facets)
| ID | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| **`1`** | Boundary | Unified Tissue-CSF Interface | Default (if `--separate-interfaces` is not used) |
| **`2`** | Boundary | Skull | Outer boundary |
| **`3`** | Boundary | Spinal Canal | Bottom fluid boundary |
| **`4`** | Boundary | Spinal Cord | Bottom porous boundary |
| **`5`** | Boundary | Aqueduct | Internal interface for flow computation |
| **`11`** | Boundary | Pia Membrane | Used if `--separate-interfaces` is set |
| **`12`** | Boundary | Ependyma | Used if `--separate-interfaces` is set |