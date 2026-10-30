from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer
import dolfinx
from dolfinx.io import XDMFFile
from mpi4py import MPI

from generateMesh import (
    POROUS_ID,
    FLUID_ID,
    INTERFACE_ID,
    PIA_ID,
    EPENDYMA_ID,
    SV_ID,
    SKULL_ID,
    SPINAL_CANAL_ID,
    SPINAL_CORD_ID,
)

app = typer.Typer(
    help="CLI for validating idealized brain FEniCSx meshes: cell quality, "
    "orientation, and marker coverage."
)

# subdomains_ftetwild labels, matching the remapping done in generateMesh.py's
# `mesh` command (raw fTetWild labels 1-4 after remapping, not the original
# fTetWild CSG-tree labels 1-6).
FTETWILD_SAS = 1
FTETWILD_PARENCHYMA = 2
FTETWILD_LV = 3
FTETWILD_AQUEDUCT = 4
FTETWILD_LABELS = {
    FTETWILD_SAS: "SAS fluid",
    FTETWILD_PARENCHYMA: "Parenchyma (porous)",
    FTETWILD_LV: "Lateral ventricles",
    FTETWILD_AQUEDUCT: "Aqueduct",
}

SUBDOMAIN_LABELS = {
    POROUS_ID: "Porous (parenchyma)",
    FLUID_ID: "Fluid (SAS + ventricles)",
}
BOUNDARY_LABELS = {
    INTERFACE_ID: "Tissue-CSF interface (unified)",
    SKULL_ID: "Skull",
    SPINAL_CANAL_ID: "Spinal canal",
    SPINAL_CORD_ID: "Spinal cord",
}
BOUNDARY_SPLIT_LABELS = {
    SKULL_ID: "Skull",
    SPINAL_CANAL_ID: "Spinal canal",
    SPINAL_CORD_ID: "Spinal cord",
    PIA_ID: "Pia membrane",
    EPENDYMA_ID: "Ependyma",
    SV_ID: "SAS-ventricle connection",
}
# Boundary IDs in `boundaries`/`boundaries_split` expected to lie on the
# exterior of the domain (as opposed to internal interfaces).
EXTERIOR_BOUNDARY_IDS = {SKULL_ID, SPINAL_CANAL_ID, SPINAL_CORD_ID}

# Expected pairs of subdomains_ftetwild values on either side of each
# internal interface tag in `boundaries_split`.
BOUNDARY_SPLIT_EXPECTED_PAIRS = {
    PIA_ID: [{FTETWILD_SAS, FTETWILD_PARENCHYMA}],
    EPENDYMA_ID: [
        {FTETWILD_PARENCHYMA, FTETWILD_AQUEDUCT},
        {FTETWILD_PARENCHYMA, FTETWILD_LV},
    ],
    SV_ID: [
        {FTETWILD_SAS, FTETWILD_AQUEDUCT},
        {FTETWILD_SAS, FTETWILD_LV},
    ],
}
# Expected pair of `subdomains` values across the unified interface.
INTERFACE_EXPECTED_PAIR = {POROUS_ID, FLUID_ID}


class Report:
    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def ok(self) -> bool:
        return len(self.errors) == 0

    def print(self) -> None:
        typer.secho(f"\n=== {self.name} ===", bold=True)
        for msg in self.infos:
            typer.echo(f"  [info] {msg}")
        for msg in self.warnings:
            typer.secho(f"  [warn] {msg}", fg=typer.colors.YELLOW)
        for msg in self.errors:
            typer.secho(f"  [FAIL] {msg}", fg=typer.colors.RED)
        if self.ok():
            typer.secho(
                f"  PASS ({len(self.warnings)} warning(s))"
                if self.warnings
                else "  PASS",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(f"  FAIL ({len(self.errors)} error(s))", fg=typer.colors.RED)


def read_all_tags(mesh_file: Path):
    """Reads the mesh and all known cell/facet tags, tolerating missing ones."""
    cell_tag_names = ["subdomains", "subdomains_ftetwild"]
    facet_tag_names = ["boundaries", "boundaries_split"]

    cell_tags: dict[str, dolfinx.mesh.MeshTags] = {}
    facet_tags: dict[str, dolfinx.mesh.MeshTags] = {}

    with XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        mesh = xdmf.read_mesh(dolfinx.cpp.mesh.GhostMode.none)

        for name in cell_tag_names:
            try:
                cell_tags[name] = xdmf.read_meshtags(mesh, name=name)
            except RuntimeError:
                pass

        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_connectivity(fdim, tdim)

        for name in facet_tag_names:
            try:
                facet_tags[name] = xdmf.read_meshtags(mesh, name=name)
            except RuntimeError:
                pass

    return mesh, cell_tags, facet_tags


def scatter_tag_values(tag: dolfinx.mesh.MeshTags, size: int) -> np.ndarray:
    """Builds a dense array of tag values indexed by entity index, -1 where untagged."""
    arr = np.full(size, -1, dtype=np.int64)
    arr[tag.indices] = tag.values
    return arr


def check_geometry(
    mesh: dolfinx.mesh.Mesh,
    report: Report,
    min_scaled_jacobian: float,
    max_radius_ratio: float,
) -> None:
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    num_vertices = mesh.topology.index_map(0).size_local

    conn = mesh.geometry.dofmap
    pts = mesh.geometry.x

    if not np.all(np.isfinite(pts)):
        report.error("Mesh contains non-finite (NaN/Inf) vertex coordinates.")
        return

    referenced = np.unique(conn)
    unused = np.setdiff1d(np.arange(num_vertices), referenced)
    if len(unused) > 0:
        report.warn(f"{len(unused)} vertex/vertices are not referenced by any cell.")

    # Tetrahedron volumes. Local vertex ordering (and hence the sign of the
    # scalar triple product) is not meaningful for straight-sided tets in
    # dolfinx: assembly always uses abs(det(J)), and dolfinx.mesh.refine
    # routinely produces child cells with flipped local ordering. Only the
    # magnitude indicates a degenerate cell.
    p0, p1, p2, p3 = (pts[conn[:, i]] for i in range(4))
    signed_vol = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0
    abs_vol = np.abs(signed_vol)
    total_volume = abs_vol.sum()

    degenerate = abs_vol < 1e-14 * max(total_volume / max(num_cells, 1), 1e-30)
    if np.any(degenerate):
        report.error(
            f"{degenerate.sum()} cell(s) have (near-)zero volume, e.g. cell "
            f"{int(np.flatnonzero(degenerate)[0])}."
        )

    report.info(f"Total mesh volume: {total_volume:.6e}")

    # Cell quality via VTK's verdict metrics.
    cells = np.hstack(
        [np.full((num_cells, 1), 4, dtype=np.int64), conn.astype(np.int64)]
    ).flatten()
    celltypes = np.full(num_cells, pv.CellType.TETRA)
    grid = pv.UnstructuredGrid(cells, celltypes, pts)
    quality = grid.cell_quality(["scaled_jacobian", "radius_ratio"])
    # Sign of scaled_jacobian mirrors local vertex ordering (see note above on
    # signed_vol) and is not meaningful here; use the magnitude.
    scaled_jacobian = np.abs(quality.cell_data["scaled_jacobian"])
    radius_ratio = quality.cell_data["radius_ratio"]

    poor_jacobian = scaled_jacobian < min_scaled_jacobian
    if np.any(poor_jacobian):
        report.warn(
            f"{poor_jacobian.sum()} cell(s) have |scaled Jacobian| below "
            f"{min_scaled_jacobian} (min={scaled_jacobian.min():.4g}); "
            "these are poorly shaped but valid."
        )

    poor_ratio = radius_ratio > max_radius_ratio
    if np.any(poor_ratio):
        report.warn(
            f"{poor_ratio.sum()} cell(s) have radius ratio above "
            f"{max_radius_ratio} (max={radius_ratio.max():.4g}); slivers likely."
        )

    report.info(
        f"|scaled_jacobian| range: [{scaled_jacobian.min():.4g}, "
        f"{scaled_jacobian.max():.4g}]; radius_ratio range: "
        f"[{radius_ratio.min():.4g}, {radius_ratio.max():.4g}]"
    )


def check_overconstrained_cells(mesh: dolfinx.mesh.Mesh, report: Report) -> None:
    """Flags cells with 3 or more of their 4 faces on the mesh boundary.

    These should have been removed by fixOverconstrainedCells.py on the base
    mesh before any refinement.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    num_cells = mesh.topology.index_map(tdim).size_local

    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)
    num_facets = mesh.topology.index_map(fdim).size_local

    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    is_exterior = np.zeros(num_facets, dtype=bool)
    is_exterior[exterior_facets] = True

    facets_per_cell = c_to_f.array.reshape(num_cells, 4)
    exterior_count = is_exterior[facets_per_cell].sum(axis=1)

    overconstrained = exterior_count >= 3
    n_bad = int(overconstrained.sum())
    if n_bad > 0:
        report.error(
            f"{n_bad} overconstrained cell(s) found (>=3 of 4 faces on the "
            "boundary). Run fixOverconstrainedCells.py before using this mesh."
        )
    else:
        report.info("No overconstrained cells found.")


def check_cell_tags(
    mesh: dolfinx.mesh.Mesh, cell_tags: dict[str, dolfinx.mesh.MeshTags], report: Report
) -> tuple[np.ndarray | None, np.ndarray | None]:
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local

    conn = mesh.geometry.dofmap
    pts = mesh.geometry.x
    p0, p1, p2, p3 = (pts[conn[:, i]] for i in range(4))
    cell_volume = np.abs(np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    subdomains_arr = None
    ftetwild_arr = None

    if "subdomains" not in cell_tags:
        report.error("Missing cell tag 'subdomains'.")
    else:
        tag = cell_tags["subdomains"]
        if len(tag.indices) != num_cells:
            report.error(
                f"'subdomains' tags {len(tag.indices)}/{num_cells} cells "
                "(some cells are untagged)."
            )
        unexpected = set(np.unique(tag.values)) - set(SUBDOMAIN_LABELS)
        if unexpected:
            report.error(f"'subdomains' has unexpected value(s): {sorted(unexpected)}.")
        subdomains_arr = scatter_tag_values(tag, num_cells)
        for value, label in SUBDOMAIN_LABELS.items():
            mask = subdomains_arr == value
            count = int(mask.sum())
            if count == 0:
                report.error(f"'subdomains' has no cells tagged {value} ({label}).")
            else:
                report.info(
                    f"subdomains={value} ({label}): {count} cells, "
                    f"volume={cell_volume[mask].sum():.6e}"
                )

    if "subdomains_ftetwild" not in cell_tags:
        report.warn("Missing cell tag 'subdomains_ftetwild'.")
    else:
        tag = cell_tags["subdomains_ftetwild"]
        if len(tag.indices) != num_cells:
            report.error(
                f"'subdomains_ftetwild' tags {len(tag.indices)}/{num_cells} cells "
                "(some cells are untagged)."
            )
        unexpected = set(np.unique(tag.values)) - set(FTETWILD_LABELS)
        if unexpected:
            report.error(
                f"'subdomains_ftetwild' has unexpected value(s): {sorted(unexpected)}."
            )
        ftetwild_arr = scatter_tag_values(tag, num_cells)
        for value, label in FTETWILD_LABELS.items():
            count = int((ftetwild_arr == value).sum())
            if count == 0:
                report.error(
                    f"'subdomains_ftetwild' has no cells tagged {value} ({label})."
                )
            else:
                report.info(f"subdomains_ftetwild={value} ({label}): {count} cells")

    if subdomains_arr is not None and ftetwild_arr is not None:
        expected_porous = ftetwild_arr == FTETWILD_PARENCHYMA
        actual_porous = subdomains_arr == POROUS_ID
        mismatch = expected_porous != actual_porous
        n_mismatch = int(mismatch.sum())
        if n_mismatch > 0:
            report.error(
                f"{n_mismatch} cell(s) disagree between 'subdomains' and "
                "'subdomains_ftetwild' porous/fluid classification."
            )

    return subdomains_arr, ftetwild_arr


def check_facet_tag(
    mesh: dolfinx.mesh.Mesh,
    tag_name: str,
    tag: dolfinx.mesh.MeshTags,
    labels: dict[int, str],
    exterior_ids: set[int],
    expected_pairs: dict[int, list[set[int]]],
    domain_arr: np.ndarray | None,
    domain_tag_name: str,
    report: Report,
) -> None:
    tdim = mesh.topology.dim
    fdim = tdim - 1
    num_facets = mesh.topology.index_map(fdim).size_local
    mesh.topology.create_connectivity(fdim, tdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)

    unexpected = set(np.unique(tag.values)) - set(labels)
    if unexpected:
        report.error(f"'{tag_name}' has unexpected value(s): {sorted(unexpected)}.")

    values_by_facet = scatter_tag_values(tag, num_facets)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    is_exterior = np.zeros(num_facets, dtype=bool)
    is_exterior[exterior_facets] = True

    tagged_exterior_mask = np.isin(values_by_facet, list(exterior_ids))
    uncovered_exterior = np.sum(is_exterior & ~tagged_exterior_mask)
    if uncovered_exterior > 0:
        report.error(
            f"'{tag_name}': {uncovered_exterior} exterior facet(s) are not tagged "
            f"with any of {sorted(exterior_ids)}."
        )

    misplaced_exterior_tag = np.sum(tagged_exterior_mask & ~is_exterior)
    if misplaced_exterior_tag > 0:
        report.error(
            f"'{tag_name}': {misplaced_exterior_tag} facet(s) tagged as exterior "
            "boundary are not actually on the mesh boundary."
        )

    for value, label in labels.items():
        count = int((values_by_facet == value).sum())
        if count == 0:
            report.error(f"'{tag_name}'={value} ({label}) has no tagged facets.")
        else:
            report.info(f"{tag_name}={value} ({label}): {count} facets")

    # Internal interfaces (anything not an exterior boundary ID) must connect
    # exactly two cells with the expected pair of domain-tag values.
    for value, pairs in expected_pairs.items():
        if value not in labels:
            continue
        facet_idx = tag.indices[tag.values == value]
        n_not_interior = 0
        n_wrong_pair = 0
        for f in facet_idx:
            cells = f_to_c.links(f)
            if len(cells) != 2:
                n_not_interior += 1
                continue
            if domain_arr is None:
                continue
            pair = {int(domain_arr[cells[0]]), int(domain_arr[cells[1]])}
            if pair not in pairs:
                n_wrong_pair += 1
        if n_not_interior > 0:
            report.error(
                f"'{tag_name}'={value} ({labels[value]}): {n_not_interior} facet(s) "
                "are not internal (don't connect exactly two cells)."
            )
        if n_wrong_pair > 0:
            report.error(
                f"'{tag_name}'={value} ({labels[value]}): {n_wrong_pair} facet(s) "
                f"connect an unexpected pair of '{domain_tag_name}' values "
                f"(expected one of {pairs})."
            )


def check_mesh(
    mesh_dir: Path,
    min_scaled_jacobian: float,
    max_radius_ratio: float,
) -> Report:
    assert MPI.COMM_WORLD.size == 1, "checkMesh.py must be run in serial."

    mesh_file = mesh_dir / f"{mesh_dir.name}.xdmf"
    report = Report(str(mesh_dir))

    if not mesh_file.exists():
        report.error(f"Mesh file not found: {mesh_file}")
        return report

    mesh, cell_tags, facet_tags = read_all_tags(mesh_file)

    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    num_vertices = mesh.topology.index_map(0).size_local
    report.info(f"{num_cells} cells, {num_vertices} vertices.")

    check_geometry(mesh, report, min_scaled_jacobian, max_radius_ratio)
    check_overconstrained_cells(mesh, report)
    subdomains_arr, ftetwild_arr = check_cell_tags(mesh, cell_tags, report)

    if "boundaries" not in facet_tags:
        report.error("Missing facet tag 'boundaries'.")
    else:
        check_facet_tag(
            mesh,
            "boundaries",
            facet_tags["boundaries"],
            BOUNDARY_LABELS,
            EXTERIOR_BOUNDARY_IDS,
            {INTERFACE_ID: [INTERFACE_EXPECTED_PAIR]},
            subdomains_arr,
            "subdomains",
            report,
        )

    if "boundaries_split" not in facet_tags:
        report.error("Missing facet tag 'boundaries_split'.")
    else:
        check_facet_tag(
            mesh,
            "boundaries_split",
            facet_tags["boundaries_split"],
            BOUNDARY_SPLIT_LABELS,
            EXTERIOR_BOUNDARY_IDS,
            BOUNDARY_SPLIT_EXPECTED_PAIRS,
            ftetwild_arr,
            "subdomains_ftetwild",
            report,
        )

    return report


@app.command()
def check(
    mesh_dir: Annotated[
        Path,
        typer.Argument(help="Mesh directory, e.g. meshes/idealized (expects idealized.xdmf inside)."),
    ],
    min_scaled_jacobian: Annotated[
        float, typer.Option(help="Warn if a cell's scaled Jacobian falls below this.")
    ] = 0.05,
    max_radius_ratio: Annotated[
        float, typer.Option(help="Warn if a cell's radius ratio exceeds this.")
    ] = 15.0,
):
    """Checks a single mesh directory for bad cells and marker consistency."""
    report = check_mesh(mesh_dir, min_scaled_jacobian, max_radius_ratio)
    report.print()
    if not report.ok():
        raise typer.Exit(code=1)


@app.command(name="check-all")
def check_all(
    root: Annotated[
        Path, typer.Option(help="Directory containing one subdirectory per mesh.")
    ] = Path("meshes"),
    min_scaled_jacobian: Annotated[
        float, typer.Option(help="Warn if a cell's scaled Jacobian falls below this.")
    ] = 0.05,
    max_radius_ratio: Annotated[
        float, typer.Option(help="Warn if a cell's radius ratio exceeds this.")
    ] = 15.0,
):
    """Checks every mesh directory found under `root`."""
    mesh_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / f"{d.name}.xdmf").exists()
    )
    if not mesh_dirs:
        typer.secho(f"No mesh directories found under {root}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    reports = [check_mesh(d, min_scaled_jacobian, max_radius_ratio) for d in mesh_dirs]
    for report in reports:
        report.print()

    n_fail = sum(1 for r in reports if not r.ok())
    typer.echo(f"\n{len(reports) - n_fail}/{len(reports)} mesh(es) passed.")
    if n_fail > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
