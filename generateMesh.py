import json
from pathlib import Path

import numpy as np
import pyvista as pv
import typer

app = typer.Typer(help="CLI for generating idealized brain meshes and FEniCSx tags.")

# Global domain/boundary IDs
POROUS_ID = 1
FLUID_ID = 2

# Boundary IDs
INTERFACE_ID = 1
PIA_ID = 11
EPENDYMA_ID = 12
SV_ID = 13
SKULL_ID = 2
SPINAL_CANAL_ID = 3
SPINAL_CORD_ID = 4
AQUEDUCT_ID = 5


@app.command()
def surfaces(
    output_dir: Path = typer.Option(
        Path("surfaces"), help="Directory to save STL files."
    ),
    show_plot: bool = typer.Option(
        False, "--show", help="Display the PyVista 3D plot before saving."
    ),
):
    """Generates the surface STLs required for the CSG tree."""
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating surface meshes...")
    skull = pv.Sphere(radius=0.08)
    parenchyma = pv.Sphere(radius=0.07)
    ventricle = pv.Sphere(radius=0.02)
    canal = pv.Cylinder(
        center=(0, 0, -0.105), direction=(0, 0, -1), radius=0.025, height=0.08
    ).triangulate()
    cord = pv.Cylinder(
        center=(0, 0, -0.105), direction=(0, 0, -1), radius=0.017, height=0.08
    ).triangulate()

    orig_center = np.array([0.0, 0.03, -0.03])
    direction = np.array([0.0, 1.0, -1.0])
    direction_norm = direction / np.linalg.norm(direction)

    # Calculate distance to parenchyma surface
    orig_dist = np.linalg.norm(orig_center)
    v4_height = 0.07 - orig_dist
    shift_v4 = (v4_height / 2.0) * direction_norm
    aqueduct_v4 = pv.Cylinder(
        center=(orig_center + shift_v4).tolist(),
        direction=direction.tolist(),
        radius=0.004,
        height=v4_height,
    ).triangulate()

    v3_height = 0.03
    shift_v3 = (v3_height / 2.0) * direction_norm
    
    aqueduct_v3 = pv.Cylinder(
        center=(orig_center - shift_v3).tolist(),
        direction=direction.tolist(),
        radius=0.004,
        height=v3_height,
    ).triangulate()

    if show_plot:
        print("Opening PyVista plot window...")
        pl = pv.Plotter()
        pl.add_mesh(parenchyma, opacity=0.6, color="blue")
        pl.add_mesh(ventricle, color="red")
        pl.add_mesh(aqueduct_v3, opacity=0.8, color="orange")
        pl.add_mesh(aqueduct_v4, opacity=0.8, color="red")
        pl.add_mesh(skull, opacity=0.2)
        pl.add_mesh(canal, opacity=0.2)
        pl.add_mesh(cord, opacity=0.7, color="blue")
        pl.background_color = "white"
        pl.show()

    print(f"Saving STLs to {output_dir}...")
    parenchyma.save(output_dir / "parenchyma.stl")
    skull.save(output_dir / "skull.stl")
    cord.save(output_dir / "cord.stl")
    canal.save(output_dir / "canal.stl")
    ventricle.save(output_dir / "ventricle.stl")
    aqueduct_v3.save(output_dir / "aqueduct_v3.stl")
    aqueduct_v4.save(output_dir / "aqueduct_v4.stl")
    print("Surfaces generated successfully.")


@app.command()
def mesh(
    stl_dir: Path = typer.Option(
        Path("surfaces"), help="Directory containing input STLs."
    ),
    name: Path = typer.Option(
        Path("mesh_out"), help="Name for saved FEniCSx XDMF files."
    ),
):
    """Generates the volumetric mesh using fTetWild and tags boundaries for FEniCSx."""
    import wildmeshing as wm

    output_dir = Path("meshes") / name
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Volumetric meshing with fTetWild
    print("Tetrahedralizing CSG tree with fTetWild...")
    tetra = wm.Tetrahedralizer(epsilon=0.002, edge_length_r=0.05, coarsen=False)

    csg_dict = {
        "operation": "union",
        "right": {
            "operation": "difference",
            "left": {
                "operation": "union",
                "left": str(stl_dir / "parenchyma.stl"),
                "right": str(stl_dir / "cord.stl"),
            },
            "right": {
                "operation": "union",
                "left": {
                    "operation": "union",
                    "left": str(stl_dir / "aqueduct_v4.stl"),
                    "right": str(stl_dir / "aqueduct_v3.stl"),
                },
                "right": str(stl_dir / "ventricle.stl"),
            },
        },
        "left": {
            "operation": "union",
            "left": str(stl_dir / "skull.stl"),
            "right": str(stl_dir / "canal.stl"),
        },
    }

    def extract_paths(d):
        paths = []
        for key, value in d.items():
            if isinstance(value, dict):
                paths.extend(extract_paths(value))
            elif isinstance(value, str) and value.endswith(".stl"):
                paths.append(Path(value))
        return paths

    missing_files = [f for f in extract_paths(csg_dict) if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required STLs: {missing_files}. Run 'surfaces' first."
        )

    tetra.load_csg_tree(json.dumps(csg_dict))
    tetra.tetrahedralize()
    point_array, cell_array, marker = tetra.get_tet_mesh()

    import basix
    import dolfinx
    import ufl
    from dolfinx.io import XDMFFile
    from mpi4py import MPI

    # 2. Re-map fTetWild markers
    print("Mapping subdomains via original np.isin logic...")
    raw_markers = np.copy(marker).flatten()
    subdomains = np.copy(raw_markers)

    subdomains[np.isin(raw_markers, [1, 2])] = 1
    subdomains[np.isin(raw_markers, [3, 4])] = 2  # parenchyma parts

    subdomains[np.isin(raw_markers, [5])] = 4  # V4
    subdomains[np.isin(raw_markers, [6])] = 5  # V3
    subdomains[np.isin(raw_markers, [7])] = 6  # LV

    labels = np.copy(subdomains)  # ftetwild labels 1-6

    subdomains[np.isin(subdomains, [2])] = 100  # tmp to avoid conflict
    subdomains[np.isin(subdomains, [1, 4, 5, 6])] = FLUID_ID
    subdomains[np.isin(subdomains, [100])] = POROUS_ID

    # 3. Create FEniCSx mesh
    print("Constructing FEniCSx mesh...")
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_WORLD,
        cells=cell_array.astype(np.int64),
        x=point_array,
        e=ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )

    # Prepare cell tags
    def create_cell_tags(mesh, values_array):
        local_entities, local_values = dolfinx.io.distribute_entity_data(
            mesh,
            mesh.topology.dim,
            cell_array.astype(np.int64),
            values_array.astype(np.int32),
        )
        adj = dolfinx.graph.adjacencylist(local_entities)
        return dolfinx.mesh.meshtags_from_entities(
            mesh, mesh.topology.dim, adj, local_values.astype(np.int32, copy=False)
        )

    ct = create_cell_tags(domain, subdomains)
    ct2 = create_cell_tags(domain, labels)

    # 4. Boundary and interface marking
    print("Marking boundaries and interfaces...")
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_entities(fdim)

    num_facets = domain.topology.index_map(fdim).size_local
    marker_values = np.zeros(num_facets, dtype=np.int32)

    def get_internal_interface_facets(cell_tags, doms=None):
        domain.topology.create_connectivity(fdim, tdim)
        f_to_c = domain.topology.connectivity(fdim, tdim)
        internal_facets = []
        for f in range(num_facets):
            cells = f_to_c.links(f)
            if len(cells) == 2:
                domains = {cell_tags.values[cells[0]], cell_tags.values[cells[1]]}
                if doms is None or set(doms) == domains:
                    internal_facets.append(f)
        return np.array(internal_facets, dtype=np.int32)

    def get_external_boundary_facets(cell_tags, subdomain_ids):
        domain.topology.create_connectivity(fdim, tdim)
        f_to_c = domain.topology.connectivity(fdim, tdim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
        target_facets = []
        for f in boundary_facets:
            c = f_to_c.links(f)[0]
            if cell_tags.values[c] in subdomain_ids:
                target_facets.append(f)
        return np.array(target_facets, dtype=np.int32)

    # Define aqueduct as interface between V4 and V3
    aqueduct_facets = get_internal_interface_facets(ct2, doms=[4, 5])

    z_min = np.min(domain.geometry.x[:, 2])
    outer_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_min, atol=0.001)
    )
    spinal_cord_facets = get_external_boundary_facets(ct, [POROUS_ID])

    # Initialize two separate arrays for the boundary tags
    marker_values_unified = np.zeros_like(marker_values)
    marker_values_split = np.zeros_like(marker_values)

    print("Computing unified interface tags...")
    tissue_csf_facets = get_internal_interface_facets(ct, doms=[1, 2])
    marker_values_unified[tissue_csf_facets] = INTERFACE_ID

    print("Computing separate tags for Pia (11), Ependyma (12), and SV (13)...")
    pia_facets = get_internal_interface_facets(ct2, doms=[1, 2])
    
    ependyma_facets_1 = get_internal_interface_facets(ct2, doms=[2, 4])
    ependyma_facets_2 = get_internal_interface_facets(ct2, doms=[2, 5])
    ependyma_facets_3 = get_internal_interface_facets(ct2, doms=[2, 6])
    ependyma_facets = np.concatenate(
        [ependyma_facets_1, ependyma_facets_2, ependyma_facets_3]
    )
    
    # SV (Foramina): Interface between SAS fluid (1) and Ventricles (4,5,6)
    sv_facets_1 = get_internal_interface_facets(ct2, doms=[1, 4])
    sv_facets_2 = get_internal_interface_facets(ct2, doms=[1, 5])
    sv_facets_3 = get_internal_interface_facets(ct2, doms=[1, 6])
    sv_facets = np.concatenate([sv_facets_1, sv_facets_2, sv_facets_3])

    # Assign split IDs
    marker_values_split[pia_facets] = PIA_ID
    marker_values_split[ependyma_facets] = EPENDYMA_ID
    marker_values_split[sv_facets] = SV_ID

    # Apply common markers to BOTH tagging arrays
    for mv in (marker_values_unified, marker_values_split):
        mv[aqueduct_facets] = AQUEDUCT_ID
        mv[outer_facets] = SKULL_ID
        mv[bottom_facets] = SPINAL_CANAL_ID
        mv[spinal_cord_facets] = SPINAL_CORD_ID

    # Filter unmarked facets and create MeshTags for unified boundaries
    idx_uni = np.where(marker_values_unified != 0)[0].astype(np.int32)
    bm_unified = dolfinx.mesh.meshtags(domain, fdim, idx_uni, marker_values_unified[idx_uni])
    bm_unified.name = "boundaries"

    # Filter unmarked facets and create MeshTags for split boundaries
    idx_split = np.where(marker_values_split != 0)[0].astype(np.int32)
    bm_split = dolfinx.mesh.meshtags(domain, fdim, idx_split, marker_values_split[idx_split])
    bm_split.name = "boundaries_split"

    # 5. Export mesh to XDMF
    print("Exporting mesh and mesh tags to XDMF...")

    ct.name = "subdomains"
    ct2.name = "subdomains_ftetwild"

    meshfile = (output_dir / name).with_suffix(".xdmf")

    with XDMFFile(domain.comm, meshfile, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(ct, domain.geometry)
        xdmf.write_meshtags(ct2, domain.geometry)
        xdmf.write_meshtags(bm_unified, domain.geometry)
        xdmf.write_meshtags(bm_split, domain.geometry)

    print(f"Number of cells: {domain.topology.index_map(tdim).size_local}")
    print(f"Meshing complete. Mesh written to {output_dir.absolute()}.")


if __name__ == "__main__":
    app()
