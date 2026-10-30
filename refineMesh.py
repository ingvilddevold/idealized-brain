import shutil
from pathlib import Path

import dolfinx
import numpy as np
import typer
from mpi4py import MPI

app = typer.Typer(
    help="CLI for iteratively refining FEniCSx meshes and transferring tags."
)


def refine_subdomain(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dict[str, dolfinx.mesh.MeshTags],
    facet_tags: dict[str, dolfinx.mesh.MeshTags],
    tag_name: str,
    subdomain_id: int,
) -> tuple[dolfinx.mesh.Mesh, dict[str, dolfinx.mesh.MeshTags], dict[str, dolfinx.mesh.MeshTags]]:
    """Refines all edges of the cells tagged `subdomain_id` in `cell_tags[tag_name]`
    and transfers every cell and facet tag to the refined mesh."""
    tdim = mesh.topology.dim
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(tdim, 1)
    c_to_e = mesh.topology.connectivity(tdim, 1)

    ct = cell_tags[tag_name]
    cells = ct.indices[ct.values == subdomain_id]
    edges = np.unique(np.hstack([c_to_e.links(c) for c in cells])).astype(np.int32)
    print(f"Refining {len(edges)} edges in subdomain {subdomain_id} of '{tag_name}'...")

    refined, parent_cells, parent_facets = dolfinx.mesh.refine(
        mesh, edges=edges, option=dolfinx.mesh.RefinementOption.parent_cell_and_facet
    )
    refined.topology.create_entities(tdim - 1)

    def transfer(tags, *parents):
        new = {}
        for name, tag in tags.items():
            new[name] = dolfinx.mesh.transfer_meshtag(tag, refined, *parents)
            new[name].name = name
        return new

    print(f"Number of cells after refinement: {refined.topology.index_map(tdim).size_global}")
    return (
        refined,
        transfer(cell_tags, parent_cells),
        transfer(facet_tags, parent_cells, parent_facets),
    )


@app.command()
def refine(
    input_dir: Path = typer.Option(
        ...,
        help="Directory containing the unrefined mesh files (e.g., meshes/idealized).",
        prompt="Path to input mesh directory",
    ),
    levels: int = typer.Option(
        2, help="Number of times to iteratively refine the mesh."
    ),
    copy_config: bool = typer.Option(
        True, "--no-config", help="Copy config.yml to the refined output directories."
    ),
):
    """
    Reads an XDMF mesh and its boundary/subdomain tags, refines them,
    and exports them to new folders for each refinement level.
    """
    mesh_file = input_dir / f"{input_dir.name}.xdmf"
    mesh_basename = mesh_file.stem
    config_file = input_dir / "config.yml"

    # Strip trailing "Ref0"
    base_name = mesh_basename.removesuffix("Ref0")

    if not mesh_file.exists():
        typer.secho(f"Error: Could not find {mesh_file}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Loading initial mesh from {input_dir}...")

    # Read the mesh and mesh tags
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        current_domain = xdmf.read_mesh()
        current_subdomains = xdmf.read_meshtags(current_domain, name="subdomains")
        current_subdomains2 = xdmf.read_meshtags(
            current_domain, name="subdomains_ftetwild"
        )

        # Create connectivity before reading facet tags
        tdim = current_domain.topology.dim
        fdim = tdim - 1
        current_domain.topology.create_connectivity(fdim, tdim)

        current_boundaries = xdmf.read_meshtags(current_domain, name="boundaries")
        current_boundaries_split = xdmf.read_meshtags(current_domain, name="boundaries_split")

    num_cells = current_domain.topology.index_map(tdim).size_local
    typer.echo(f"Num cells before refinement: {num_cells}")

    # Iterative refinement
    current_domain.topology.create_entities(1)

    for i in range(1, levels + 1):
        typer.echo(f"\n--- Starting Refinement Level {i} ---")

        # Refine domain
        domain_ref, parent_cell_ref, parent_facet_ref = dolfinx.mesh.refine(
            current_domain, option=3
        )

        # Re-establish connectivity for the newly refined mesh
        domain_ref.topology.create_connectivity(fdim, tdim)
        domain_ref.topology.create_entities(1)

        # Transfer tags
        subdomains_ref = dolfinx.mesh.transfer_meshtag(
            current_subdomains, domain_ref, parent_cell_ref, parent_facet_ref
        )
        subdomains_ref2 = dolfinx.mesh.transfer_meshtag(
            current_subdomains2, domain_ref, parent_cell_ref, parent_facet_ref
        )
        boundaries_ref = dolfinx.mesh.transfer_meshtag(
            current_boundaries,
            domain_ref,
            parent_cell_ref,
            parent_facet_ref,
        )
        boundaries_split_ref = dolfinx.mesh.transfer_meshtag(
            current_boundaries_split,
            domain_ref,
            parent_cell_ref,
            parent_facet_ref,
        )

        # Set up output directory
        out_name = f"{base_name}Ref{i}"
        out_dir = input_dir.with_name(f"{base_name}Ref{i}")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Export meshes
        subdomains_ref.name = "subdomains"
        subdomains_ref2.name = "subdomains_ftetwild"
        boundaries_ref.name = "boundaries"
        boundaries_split_ref.name = "boundaries_split"

        typer.echo(f"Writing data to {out_dir}...")
        with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, out_dir / f"{out_name}.xdmf", "w"
        ) as xdmf:
            xdmf.write_mesh(domain_ref)
            xdmf.write_meshtags(subdomains_ref, domain_ref.geometry)
            xdmf.write_meshtags(subdomains_ref2, domain_ref.geometry)
            xdmf.write_meshtags(boundaries_ref, domain_ref.geometry)
            xdmf.write_meshtags(boundaries_split_ref, domain_ref.geometry)

        # Copy config if requested and it exists
        if copy_config and config_file.exists():
            shutil.copyfile(config_file, out_dir / "config.yml")

        new_num_cells = domain_ref.topology.index_map(tdim).size_local
        typer.echo(f"Num cells after refinement {i}: {new_num_cells}")

        # Overwrite current variables to prepare for the next iteration
        current_domain = domain_ref
        current_subdomains = subdomains_ref
        current_subdomains2 = subdomains_ref2
        current_boundaries = boundaries_ref
        current_boundaries_split = boundaries_split_ref

    typer.secho("\nAll refinements completed successfully.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
