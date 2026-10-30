from pathlib import Path
from typing import Annotated

import typer

import generateMesh

app = typer.Typer(help="Run the full meshing pipeline in one go.")


@app.command()
def run(
    name: Annotated[
        str, typer.Option(help="Mesh name. Writes surfaces/<name> and meshes/<name> "
        "(meshes/<name>Ref0, Ref1, ... if --levels > 0).")
    ] = "idealized",
    sas_thickness_mm: Annotated[float, typer.Option(help="SAS thickness (mm).")] = 10.0,
    fillet_radius_mm: Annotated[
        float, typer.Option(help="Cord/parenchyma fillet radius (mm). 0 is sharp.")
    ] = 15.0,
    aqueduct_fillet_radius_mm: Annotated[
        float, typer.Option(help="Aqueduct/ventricle fillet radius (mm). 0 is sharp.")
    ] = 2.0,
    sphere_resolution: Annotated[
        int, typer.Option(help="Theta/phi resolution for the spheres.")
    ] = 100,
    sas_cells_across: Annotated[
        float,
        typer.Option(
            help="Target number of cells across the SAS, via an fTetWild sizing "
            "field. 0 instead meshes uniformly with --fineness and then refines "
            "the SAS once (which does not refine the curved geometry)."
        ),
    ] = 2,
    aqueduct_edge_mm: Annotated[
        float, typer.Option(help="Edge length (mm) in the aqueduct.")
    ] = 2.0,
    coarse_edge_mm: Annotated[
        float, typer.Option(help="Edge length (mm) away from the SAS and aqueduct.")
    ] = 8.0,
    grading: Annotated[
        float, typer.Option(help="Edge length growth per unit distance from the SAS/aqueduct.")
    ] = 0.3,
    epsilon: Annotated[
        float, typer.Option(help="fTetWild envelope, relative to the bounding box diagonal.")
    ] = generateMesh.FTETWILD_EPSILON,
    fineness: Annotated[
        float, typer.Option(help="fTetWild edge_length_r, if --sas-cells-across 0.")
    ] = 0.02,
    levels: Annotated[
        int, typer.Option(help="Number of uniform refinement levels after the fix.")
    ] = 0,
    check: Annotated[
        bool, typer.Option(help="Run checkMesh on every generated mesh.")
    ] = True,
):
    """Surfaces -> fTetWild mesh -> SAS refinement -> fix overconstrained cells
    -> uniform refinement -> check."""
    stl_dir = Path("surfaces") / name
    # Only use the RefN suffix when there are refinement levels to tell apart
    mesh_names = [f"{name}Ref{i}" for i in range(levels + 1)] if levels > 0 else [name]
    base = mesh_names[0]
    base_dir = Path("meshes") / base
    mesh_file = base_dir / f"{base}.xdmf"

    typer.secho("\n[1/6] Surfaces", bold=True)
    generateMesh.surfaces(
        output_dir=stl_dir,
        sas_thickness_mm=sas_thickness_mm,
        sphere_resolution=sphere_resolution,
        fillet_radius_mm=fillet_radius_mm,
        aqueduct_fillet_radius_mm=aqueduct_fillet_radius_mm,
    )

    typer.secho("\n[2/6] Volume mesh", bold=True)
    sas_edge_mm = sas_thickness_mm / sas_cells_across if sas_cells_across > 0 else None
    generateMesh.mesh(
        stl_dir=stl_dir,
        name=Path(base),
        fineness=fineness,
        sas_edge_mm=sas_edge_mm,
        aqueduct_edge_mm=aqueduct_edge_mm,
        coarse_edge_mm=coarse_edge_mm,
        grading=grading,
        epsilon=epsilon,
    )

    # Imported only now: loading dolfinx before wildmeshing has run crashes
    # fTetWild (clashing spdlog libraries)
    import checkMesh
    import fixOverconstrainedCells
    import refineMesh

    tag_names = ["subdomains", "subdomains_ftetwild"]
    facet_names = ["boundaries", "boundaries_split"]
    mesh, cell_tags, facet_tags = fixOverconstrainedCells.read_all_tags(
        mesh_file, tag_names, facet_names
    )

    typer.secho("\n[3/6] SAS refinement", bold=True)
    if sas_edge_mm is None:
        mesh, cell_tags, facet_tags = refineMesh.refine_subdomain(
            mesh, cell_tags, facet_tags, "subdomains_ftetwild", 1
        )
    else:
        typer.echo("Skipped (SAS already resolved by the sizing field).")

    # Must happen before uniform refinement, which multiplies overconstrained cells
    typer.secho("\n[4/6] Fix overconstrained cells", bold=True)
    mesh, cell_tags, facet_tags = fixOverconstrainedCells.fix_all_overconstrained_cells(
        mesh, cell_tags, facet_tags
    )
    fixOverconstrainedCells.write_all_tags(mesh_file, mesh, cell_tags, facet_tags)

    typer.secho("\n[5/6] Uniform refinement", bold=True)
    if levels > 0:
        refineMesh.refine(input_dir=base_dir, levels=levels, copy_config=True)
    else:
        typer.echo("Skipped (levels=0).")

    typer.secho("\n[6/6] Check", bold=True)
    if not check:
        typer.echo("Skipped.")
        return
    reports = [checkMesh.check_mesh(Path("meshes") / n, 0.05, 15.0) for n in mesh_names]
    for report in reports:
        report.print()
    if not all(r.ok() for r in reports):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
