# Idealized brain geometries

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
### 1. Mesh Generation (generateIdealizedMesh.py)
Generates surface STLs using Pyvista and creates a volumetric FEniCSx mesh from these using CSG (Constructive Solid Geometry), also generating properly tagged subdomains and boundaries.

**Generate surface STLs:**

```bash
python generateIdealizedMesh.py surfaces --output-dir stls/
```
(Add `--show` to preview the geometry in a PyVista plot).

**Generate and tag the FEniCSx mesh:**

```bash
python generateIdealizedMesh.py mesh --stl-dir stls/ --output-dir mesh_out/
```
(Add `--separate-interfaces` to tag the pial membrane and ventricular walls (ependyma) with distinct IDs instead of a unified interface tag).

Note: It is also possible to generate the mesh from the surface STL with fTetWild directly from command line as
```bash
ftetwild --csg csg.json --output mesh/idealizedBrainMesh -e 0.002 -l 0.05
```

### 2. Mesh Refinement (refineMesh.py)
Iteratively refines an existing FEniCSx mesh while correctly transferring cell (subdomain) and facet (boundary) tags to the newly created child meshes.

**Refine a mesh:**

```bash
python refineMesh.py --input-dir mesh_out/ --levels 2
```
This will read the base mesh and sequentially generate `Ref1`, `Ref2`, etc., in sibling directories.
