import shutil
from pathlib import Path

import typer
import dolfinx
from mpi4py import MPI

app = typer.Typer(
    help="CLI for iteratively refining FEniCSx meshes and transferring tags."
)


@app.command()
def refine(
    input_dir: Path = typer.Option(
        ...,
        help="Directory containing the unrefined mesh files (e.g., meshes/idealized).",
        prompt="Path to input mesh directory",
    ),
    mesh_basename: str = typer.Option(
        "idealized",
        help="Base name of the mesh files (e.g., 'idealized' expects 'idealized.xdmf').",
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
    meshfile = (input_dir / input_dir.name).with_suffix(".xdmf")
    configfile = input_dir / "config.yml"

    if not meshfile.exists():
        typer.secho(f"Error: Could not find {meshfile}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Loading initial mesh from {input_dir}...")

    # 1. Read the mesh and mesh tags
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, meshfile, "r") as xdmf:
        current_domain = xdmf.read_mesh()
        current_subdomains = xdmf.read_meshtags(current_domain, name="subdomains")
        current_subdomains2 = xdmf.read_meshtags(current_domain, name="subdomains_ftetwild")

        # Create connectivity before reading facet tags
        tdim = current_domain.topology.dim
        fdim = tdim - 1
        current_domain.topology.create_connectivity(fdim, tdim)

        current_boundaries = xdmf.read_meshtags(current_domain, name="boundaries")

    num_cells = current_domain.topology.index_map(tdim).size_local
    typer.echo(f"Num cells before refinement: {num_cells}")

    # 3. Iterative refinement
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
            current_boundaries, domain_ref, parent_cell_ref, parent_facet_ref,
        )

        # Set up output directory
        out_name = f"{mesh_basename}Ref{i}"
        out_dir = input_dir.with_name(input_dir.name + f"Ref{i}")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Export meshes
        subdomains_ref.name = "subdomains"
        subdomains_ref2.name = "subdomains_ftetwild"
        boundaries_ref.name = "boundaries"

        typer.echo(f"Writing data to {out_dir}...")
        with dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, out_dir / f"{out_name}.xdmf", "w"
        ) as xdmf:
            xdmf.write_mesh(domain_ref)
            xdmf.write_meshtags(subdomains_ref, domain_ref.geometry)
            xdmf.write_meshtags(subdomains_ref2, domain_ref.geometry)
            xdmf.write_meshtags(boundaries_ref, domain_ref.geometry)

        # Copy config if requested and it exists
        if copy_config and configfile.exists():
            shutil.copyfile(configfile, out_dir / "config.yml")

        new_num_cells = domain_ref.topology.index_map(tdim).size_local
        typer.echo(f"Num cells after refinement {i}: {new_num_cells}")

        # Overwrite current variables to prepare for the next iteration
        current_domain = domain_ref
        current_subdomains = subdomains_ref
        current_subdomains2 = subdomains_ref2
        current_boundaries = boundaries_ref

    typer.secho("\nAll refinements completed successfully.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
