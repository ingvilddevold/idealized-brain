import json
from pathlib import Path
from typing import Annotated

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

# fTetWild envelope tolerance (relative to the bounding box diagonal)
FTETWILD_EPSILON = 0.0005


def _sphere_with_tube(
    sphere_radius: float,
    tube_radius: float,
    z_bottom: float,
    fillet_radius: float,
    resolution: int,
) -> pv.PolyData:
    """Single closed surface: a sphere at the origin with a tube of
    `tube_radius` leaving it downwards to `z_bottom`, joined by a circular
    fillet of `fillet_radius` (0 gives a sharp corner).

    Built as one surface of revolution.
    """
    S, R, rho = sphere_radius, tube_radius, fillet_radius
    # Fillet circle centre, tangent to the tube wall and to the sphere
    c = np.array([R + rho, -np.sqrt((S + rho) ** 2 - (R + rho) ** 2)])
    on_sphere = S * c / np.linalg.norm(c)

    # Sphere arc from the north pole (polar angle 0) down to the fillet
    n_arc = resolution // 2
    phi = np.linspace(0.0, np.arctan2(on_sphere[0], on_sphere[1]), n_arc)
    arc = S * np.column_stack((np.sin(phi), np.cos(phi)))

    # Fillet arc around c, from the sphere tangent point to the tube wall
    a0 = np.arctan2(-c[1], -c[0])
    a = np.linspace(a0, np.pi, max(int(n_arc * (np.pi - a0) / np.pi), 2))[1:]
    fillet = c + rho * np.column_stack((np.cos(a), np.sin(a)))

    # Tube wall down to z_bottom, then the bottom cap in to the axis. Both are
    # subdivided: as single profile segments they revolve into long needle
    # triangles, which fTetWild fails to insert (the cap then bulges out).
    h = min(S * np.pi / n_arc, 2 * np.pi * R / resolution)

    def segment(p, q):
        n = max(int(np.ceil(np.linalg.norm(np.subtract(q, p)) / h)), 1)
        return np.linspace(p, q, n + 1)[1:]

    wall = segment(fillet[-1], [R, z_bottom])
    cap = segment([R, z_bottom], [0.0, z_bottom])
    rz = np.vstack((arc, fillet, wall, cap))
    points = np.column_stack((rz[:, 0], np.zeros(len(rz)), rz[:, 1]))
    profile = pv.lines_from_points(points)
    mesh = profile.extrude_rotate(resolution=resolution, capping=False).triangulate()
    # Tolerance needed to merge the 0/360 degree seam and the poles
    mesh = mesh.clean(tolerance=1e-9, absolute=True)
    return mesh.compute_normals(auto_orient_normals=True)

def _sizing_field(
    skull_stl: Path,
    brain_stl: Path,
    ventricle_stl: Path,
    h_sas: float,
    h_aqueduct: float,
    h_coarse: float,
    grading: float = 0.3,
):
    """fTetWild sizing function: edge length h_sas in the SAS (inside the
    skull surface, outside the parenchyma surface) and h_aqueduct in the
    aqueduct (the part of the ventricle surface outside the ventricle
    sphere), each growing by `grading` per unit distance, up to h_coarse."""
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

    def implicit(stl):
        f = vtkImplicitPolyDataDistance()
        f.SetInput(pv.read(stl))
        return f

    skull, brain, ventricle = implicit(skull_stl), implicit(brain_stl), implicit(ventricle_stl)
    ventricle_radius = _ventricle_radius(ventricle_stl)

    def size(x):
        # Signed distances to each region, negative inside it
        d_sas = max(skull.EvaluateFunction(x), -brain.EvaluateFunction(x))
        d_aq = max(ventricle.EvaluateFunction(x), ventricle_radius - np.linalg.norm(x))
        return min(
            h_sas + grading * max(d_sas, 0.0),
            h_aqueduct + grading * max(d_aq, 0.0),
            h_coarse,
        )

    return size


def _ventricle_radius(ventricle_stl: Path) -> float:
    """Radius of the ventricle sphere (centred at the origin): its closest point,
    since the aqueduct and its fillet only lie further out."""
    return np.linalg.norm(pv.read(ventricle_stl).points, axis=1).min()


@app.command()
def surfaces(
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save STL files.")
    ] = Path("surfaces"),
    sas_thickness_mm: Annotated[
        float,
        typer.Option(
            help="SAS thickness (mm), i.e. the gap between parenchyma and "
            "skull. Also used for the spinal canal CSF gap."
        ),
    ] = 10.0,
    sphere_resolution: Annotated[
        int,
        typer.Option(help="Theta/phi resolution for the spheres."),
    ] = 100,
    fillet_radius_mm: Annotated[
        float,
        typer.Option(
            help="Fillet radius (mm) where the spinal cord meets the "
            "parenchyma. The canal/skull fillet is this minus the SAS "
            "thickness, so the CSF gap stays uniform. 0 gives sharp corners."
        ),
    ] = 15.0,
    aqueduct_fillet_radius_mm: Annotated[
        float,
        typer.Option(
            help="Fillet radius (mm) where the aqueduct meets the ventricle. "
            "0 gives a sharp corner."
        ),
    ] = 2.0,
    show_plot: Annotated[
        bool, typer.Option("--show", help="Display the PyVista 3D plot before saving.")
    ] = False,
):
    """Generates the surface STLs required for the CSG tree."""
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating surface meshes...")
    parenchyma_radius = 0.07
    cord_radius = 0.017
    ventricle_radius = 0.02
    sas_thickness = sas_thickness_mm * 1e-3
    skull_radius = parenchyma_radius + sas_thickness
    # Spinal canal CSF gap matches the cranial SAS gap thickness
    canal_radius = cord_radius + sas_thickness

    cord_fillet = fillet_radius_mm * 1e-3
    canal_fillet = max(cord_fillet - sas_thickness, 0.0)
    if cord_fillet > 0 and canal_fillet == 0:
        print("Warning: fillet radius <= SAS thickness, so the skull/canal corner is sharp.")

    z_bottom = -0.145
    skull = _sphere_with_tube(
        skull_radius, canal_radius, z_bottom, canal_fillet, sphere_resolution
    )
    parenchyma = _sphere_with_tube(
        parenchyma_radius, cord_radius, z_bottom, cord_fillet, sphere_resolution
    )
    # Aqueduct runs radially from the ventricle out to the parenchyma surface,
    # along (0, 1, -1). Build it along -z, then rotate into place.
    ventricle = _sphere_with_tube(
        ventricle_radius,
        0.004,
        -parenchyma_radius,
        aqueduct_fillet_radius_mm * 1e-3,
        sphere_resolution,
    ).rotate_x(45)

    if show_plot:
        print("Opening PyVista plot window...")
        pl = pv.Plotter()
        pl.add_mesh(parenchyma, opacity=0.6, color="blue")
        pl.add_mesh(ventricle, color="red")
        pl.add_mesh(skull, opacity=0.2)
        pl.background_color = "white"
        pl.show()

    print(f"Saving STLs to {output_dir}...")
    skull.save(output_dir / "skull_canal.stl")
    parenchyma.save(output_dir / "parenchyma_cord.stl")
    ventricle.save(output_dir / "ventricle_aqueduct.stl")
    print("Surfaces generated successfully.")


@app.command()
def mesh(
    stl_dir: Annotated[
        Path, typer.Option(help="Directory containing input STLs.")
    ] = Path("surfaces"),
    name: Annotated[
        Path, typer.Option(help="Name for saved FEniCSx XDMF files.")
    ] = Path("mesh_out"),
    fineness: Annotated[
        float,
        typer.Option(
            help="Target edge length as a fraction of the bounding box diagonal "
            "(fTetWild's edge_length_r). Smaller values give a finer mesh with "
            "more cells."
        ),
    ] = 0.02,
    sas_edge_mm: Annotated[
        float | None,
        typer.Option(
            help="Target edge length (mm) in the SAS. If set, a sizing field "
            "replaces the uniform --fineness: edges are this size in the SAS "
            "and grow away from it up to --coarse-edge-mm."
        ),
    ] = None,
    aqueduct_edge_mm: Annotated[
        float,
        typer.Option(help="Edge length (mm) in the aqueduct (with --sas-edge-mm)."),
    ] = 2.0,
    coarse_edge_mm: Annotated[
        float,
        typer.Option(help="Edge length (mm) away from the SAS and aqueduct (with --sas-edge-mm)."),
    ] = 8.0,
    grading: Annotated[
        float,
        typer.Option(
            help="Growth of the edge length per unit distance from the SAS and "
            "aqueduct (with --sas-edge-mm). Neighbouring cells differ in size by "
            "roughly a factor 1 + grading."
        ),
    ] = 0.3,
    epsilon: Annotated[
        float,
        typer.Option(
            help="fTetWild envelope: how far the mesh surfaces may deviate from "
            "the STLs, relative to the bounding box diagonal (~0.3 m)."
        ),
    ] = FTETWILD_EPSILON,
):
    """Generates the volumetric mesh using fTetWild and tags boundaries for FEniCSx."""
    import wildmeshing as wm

    output_dir = Path("meshes") / name
    output_dir.mkdir(exist_ok=True, parents=True)

    skull_stl = stl_dir / "skull_canal.stl"
    brain_stl = stl_dir / "parenchyma_cord.stl"
    ventricle_stl = stl_dir / "ventricle_aqueduct.stl"
    missing_files = [f for f in (skull_stl, brain_stl, ventricle_stl) if not f.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required STLs: {missing_files}. Run 'surfaces' first."
        )

    # 1. Volumetric meshing with fTetWild
    if sas_edge_mm is None:
        print(f"Tetrahedralizing CSG tree with fTetWild (fineness={fineness})...")
        edge_length_r = fineness
    else:
        print(
            f"Tetrahedralizing CSG tree with fTetWild (SAS edge {sas_edge_mm} mm, "
            f"aqueduct edge {aqueduct_edge_mm} mm, coarse edge {coarse_edge_mm} mm)..."
        )
        # edge_length_r caps the edge length, so lift it to the coarse size
        bounds = np.array(pv.read(skull_stl).bounds).reshape(3, 2)
        edge_length_r = coarse_edge_mm * 1e-3 / np.linalg.norm(np.diff(bounds))
    tetra = wm.Tetrahedralizer(
        epsilon=epsilon, edge_length_r=edge_length_r, coarsen=False
    )

    csg_dict = {
        "operation": "union",
        "left": str(skull_stl),
        "right": {
            "operation": "difference",
            "left": str(brain_stl),
            "right": str(ventricle_stl),
        },
    }
    tetra.load_csg_tree(json.dumps(csg_dict))
    if sas_edge_mm is not None:
        tetra.set_sizing_field_from_func(
            _sizing_field(
                skull_stl,
                brain_stl,
                ventricle_stl,
                sas_edge_mm * 1e-3,
                aqueduct_edge_mm * 1e-3,
                coarse_edge_mm * 1e-3,
                grading,
            )
        )
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

    # fTetWild marks each cell with the (1-based) index of the last CSG input
    # containing it: 1 SAS, 2 parenchyma, 3 ventricle + aqueduct. The latter
    # is split into LV (3) and aqueduct (4) by distance from the ventricle
    # centre (the origin).
    ventricle_radius = _ventricle_radius(ventricle_stl)
    centroid_dist = np.linalg.norm(point_array[cell_array].mean(axis=1), axis=1)
    subdomains[(raw_markers == 3) & (centroid_dist > ventricle_radius)] = 4

    labels = np.copy(subdomains)  # ftetwild labels 1-4

    subdomains[np.isin(subdomains, [2])] = 100  # tmp to avoid conflict
    subdomains[np.isin(subdomains, [1, 3, 4])] = FLUID_ID
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

    domain.topology.create_connectivity(fdim, tdim)
    # Outlet plane from the input geometry, not the mesh, so a stray vertex
    # below it cannot shift the plane
    z_min = pv.read(skull_stl).bounds[4]
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
    ependyma_facets_2 = get_internal_interface_facets(ct2, doms=[2, 3])
    ependyma_facets = np.concatenate([ependyma_facets_1, ependyma_facets_2])

    # SV (Foramina): Interface between SAS fluid (1) and Ventricles (3,4)
    sv_facets_1 = get_internal_interface_facets(ct2, doms=[1, 4])
    sv_facets_2 = get_internal_interface_facets(ct2, doms=[1, 3])
    sv_facets = np.concatenate([sv_facets_1, sv_facets_2])

    # Assign split IDs
    marker_values_split[pia_facets] = PIA_ID
    marker_values_split[ependyma_facets] = EPENDYMA_ID
    marker_values_split[sv_facets] = SV_ID

    # Apply common markers to BOTH tagging arrays
    for mv in (marker_values_unified, marker_values_split):
        mv[outer_facets] = SKULL_ID
        mv[bottom_facets] = SPINAL_CANAL_ID
        mv[spinal_cord_facets] = SPINAL_CORD_ID

    # Filter unmarked facets and create MeshTags for unified boundaries
    idx_uni = np.where(marker_values_unified != 0)[0].astype(np.int32)
    bm_unified = dolfinx.mesh.meshtags(
        domain, fdim, idx_uni, marker_values_unified[idx_uni]
    )
    bm_unified.name = "boundaries"

    # Filter unmarked facets and create MeshTags for split boundaries
    idx_split = np.where(marker_values_split != 0)[0].astype(np.int32)
    bm_split = dolfinx.mesh.meshtags(
        domain, fdim, idx_split, marker_values_split[idx_split]
    )
    bm_split.name = "boundaries_split"

    # 5. Export mesh to XDMF
    print("Exporting mesh and mesh tags to XDMF...")

    ct.name = "subdomains"
    ct2.name = "subdomains_ftetwild"

    meshfile = output_dir / f"{Path(name).name}.xdmf"

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
