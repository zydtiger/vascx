"""
Calculate Arteriole-Venular Ratios from retinal vessel masks.

This script loads arteriole and venular mask images, calculates both mean and largest diameters
restricted to multiple zones (disc, A, B, C) based on disc diameters from optic disc center, and
computes both mean and largest arteriole-venular ratios.
Outputs MAD (Mean Artery Diameter), MVD (Mean Vein Diameter), AVR (Mean Ratio),
LAD (Largest Artery Diameter), LVD (Largest Vein Diameter), and LAVR (Largest Ratio)
for each zone in separate CSV files.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from skimage import io, measure
from skimage.morphology import skeletonize
from scipy import ndimage
from typing import Optional
import typer

ZONE_PARAMS: dict[str, tuple[float, float]] = {
    "disc": (0.0, 0.5),  # disc diameter
    "A": (0.5, 1.0),
    "B": (1.0, 1.5),
    "C": (1.0, 2.5),
}

app = typer.Typer(
    help="Calculate Arteriole-Venular Ratios (AVR and LAVR) from retinal vessel masks"
)


def load_mask_image(filepath: Path) -> np.ndarray:
    """Load a mask image as a binary numpy array."""
    img = io.imread(filepath)
    return img > 0


def load_disc_mask(mask_name: str, discs_dir: Path) -> np.ndarray:
    """Load the corresponding disc mask for a given vessel mask."""
    mask_base_name = os.path.splitext(mask_name)[0]
    disc_filename = f"{mask_base_name}.png"
    disc_path = discs_dir / disc_filename
    return load_mask_image(disc_path)


def create_zone_mask(
    disc_mask: np.ndarray,
    inner_diameter_factor: float,
    outer_diameter_factor: float,
) -> np.ndarray:
    """Create a binary mask for a specified zone using disc diameter factors.

    Args:
        disc_mask: Binary mask of optic disc
        inner_diameter_factor: Inner boundary as factor of disc diameter (e.g., 0.5 = half disc diameter)
        outer_diameter_factor: Outer boundary as factor of disc diameter (e.g., 1.0 = one disc diameter)

    Returns:
        Binary mask for the specified zone
    """
    # Find disc center and radius
    labeled_disc = measure.label(disc_mask)
    regions = measure.regionprops(labeled_disc)

    # Get the largest region (should be the disc)
    largest_region = max(regions, key=lambda r: r.area)
    center_y, center_x = largest_region.centroid

    # Calculate disc radius as average of minor and major axes lengths divided by 2
    disc_diameter = (
        largest_region.major_axis_length + largest_region.minor_axis_length
    ) / 2

    # Create coordinate grids
    height, width = disc_mask.shape
    y, x = np.ogrid[:height, :width]

    # Calculate distance from center for each pixel
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    zone_inner_radius = inner_diameter_factor * disc_diameter
    zone_outer_radius = outer_diameter_factor * disc_diameter
    zone_mask = (distances >= zone_inner_radius) & (distances <= zone_outer_radius)

    return zone_mask


def calculate_mean_diameter(
    mask: np.ndarray, disc_mask: np.ndarray, zone: str
) -> float:
    """Calculate mean vessel diameter from a binary mask using distance transform and skeleton.

    Args:
        mask: Binary mask of vessels
        disc_mask: Binary mask of optic disc (required for zone restriction)
        zone: Zone to restrict analysis to (from ZONE_PARAMS)
    """
    # Create Zone mask if restriction is enabled
    zone_mask = create_zone_mask(disc_mask, ZONE_PARAMS[zone][0], ZONE_PARAMS[zone][1])
    restricted_mask = mask & zone_mask

    # Calculate distance transform
    distance = ndimage.distance_transform_edt(restricted_mask)

    # Extract skeleton (centerline) of vessels
    skeleton = skeletonize(restricted_mask > 0)
    # Get distances only from centerline pixels
    vessel_distances = distance[skeleton]
    # If empty (no skeletonized pixels), return 0.0
    if vessel_distances.size == 0:
        return 0.0
    # Diameter is 2 * radius (distance transform gives radius)
    diameters = vessel_distances * 2

    # Return mean diameter
    return np.mean(diameters)


def calculate_largest_diameter(
    mask: np.ndarray, disc_mask: np.ndarray, zone: str
) -> float:
    """Calculate largest vessel diameter from a binary mask using distance transform and skeleton.

    Args:
        mask: Binary mask of vessels
        disc_mask: Binary mask of optic disc (required for zone restriction)
        zone: Zone to restrict analysis to (from ZONE_PARAMS)
    """
    # Create zone mask
    zone_mask = create_zone_mask(disc_mask, ZONE_PARAMS[zone][0], ZONE_PARAMS[zone][1])
    restricted_mask = mask & zone_mask

    # Calculate distance transform
    distance = ndimage.distance_transform_edt(restricted_mask)

    # Extract skeleton (centerline) of vessels
    skeleton = skeletonize(restricted_mask > 0)
    # Get distances only from centerline pixels
    vessel_distances = distance[skeleton]
    # If empty (no skeletonized pixels), return 0.0
    if vessel_distances.size == 0:
        return 0.0
    # Diameter is 2 * radius (distance transform gives radius)
    diameters = vessel_distances * 2

    # Return largest diameter
    return np.max(diameters)


def get_diagnosis(mask_name: str, diagnosis_data: pd.DataFrame) -> Optional[str]:
    """Get diagnosis for a given mask name from the diagnosis dataframe."""
    mask_base_name = os.path.splitext(mask_name)[0]

    id = mask_base_name.split("_")[0]
    location = mask_base_name.split("_")[1]

    matching_row = diagnosis_data[diagnosis_data["ID"] == id]
    if len(matching_row) == 0:
        return None

    diagnosis_col = (
        "Left-Diagnostic Keywords"
        if location == "left"
        else "Right-Diagnostic Keywords"
    )

    return matching_row[diagnosis_col].iloc[0]


def process_mask_pair(
    mask_name: str,
    arteries_dir: Path,
    veins_dir: Path,
    discs_dir: Path,
    diagnosis_data: pd.DataFrame,
    zones: list[str],
) -> dict[str, dict]:
    """Process a pair of masks for multiple zones and return results for each zone.

    Returns:
        Dictionary with zone names as keys and result dictionaries as values.
        Each result dictionary contains MAD, MVD, AVR, LAD, LVD, LAVR.
    """
    # Load masks
    artery_mask = load_mask_image(arteries_dir / mask_name)
    vein_mask = load_mask_image(veins_dir / mask_name)
    disc_mask = load_disc_mask(mask_name, discs_dir)

    # Get diagnosis (same for all zones)
    diagnosis = get_diagnosis(mask_name, diagnosis_data)

    results = {}

    # Process each zone
    for zone in zones:
        # Calculate mean diameters for this zone
        MAD = calculate_mean_diameter(artery_mask, disc_mask, zone)
        MVD = calculate_mean_diameter(vein_mask, disc_mask, zone)

        # Calculate AVR (Arteriole-Venular Ratio)
        AVR = MAD / MVD if MVD != 0 else 0

        # Calculate largest diameters for this zone
        LAD = calculate_largest_diameter(artery_mask, disc_mask, zone)
        LVD = calculate_largest_diameter(vein_mask, disc_mask, zone)

        # Calculate LAVR (Largest Arteriole-Venular Ratio)
        LAVR = LAD / LVD if LVD != 0 else 0

        results[zone] = {
            "MAD": MAD,
            "MVD": MVD,
            "AVR": AVR,
            "LAD": LAD,
            "LVD": LVD,
            "LAVR": LAVR,
            "diagnosis": diagnosis,
        }

    return results


@app.command()
def main(
    arteries_dir: Path = typer.Option(
        ...,
        "--arteries",
        help="Directory containing arteriole mask images",
    ),
    veins_dir: Path = typer.Option(
        ...,
        "--veins",
        help="Directory containing venular mask images",
    ),
    discs_dir: Path = typer.Option(
        ...,
        "--discs",
        help="Directory containing optic disc mask images",
    ),
    diagnosis_path: Path = typer.Option(
        ..., "--diagnosis", help="Diagnostic data table file."
    ),
    ref_dir: Path = typer.Option(
        None, "--ref", help="Directory containing image names to be processed"
    ),
    output_dir: Path = typer.Option(
        Path(__file__).parent,
        "--output",
        help="Output directory for AVR and LAVR results CSV files",
    ),
):
    """Calculate AVR and LAVR from retinal vessel masks for multiple zones."""

    # Validate input directories
    for dir_path, dir_name in [
        (arteries_dir, "arteries"),
        (veins_dir, "veins"),
        (discs_dir, "discs"),
    ]:
        if not dir_path.exists():
            typer.echo(
                f"Error: {dir_name.capitalize()} directory '{dir_path}' not found.",
                err=True,
            )
            raise typer.Exit(1)

    # Check data completeness
    if ref_dir is None:
        ref_dir = arteries_dir
    queued_imgs = [img.name for img in ref_dir.glob("*.png")]
    for img_name in queued_imgs:
        for dir_path in [arteries_dir, veins_dir, discs_dir]:
            target_img = dir_path / img_name
            if not target_img.exists():
                typer.echo(f"Error: {target_img} not found.", err=True)
                raise typer.Exit(1)

    typer.echo(f"Found {len(queued_imgs)} queued image files.")

    # Try to load diagnosis data if available
    try:
        diagnosis_data = pd.read_csv(diagnosis_path)
        typer.echo(f"Loaded diagnosis data from {diagnosis_path}")
    except Exception as e:
        typer.echo(f"Error: Could not load {diagnosis_path}: {e}", err=True)
        raise typer.Exit(1)

    # Define zones to process
    zones_to_process = list(ZONE_PARAMS.keys())
    typer.echo(f"Processing zones: {', '.join(zones_to_process)}")

    # Initialize results dictionary for each zone
    results_by_zone = {zone: [] for zone in zones_to_process}

    # Process each matching mask pair for all zones
    for mask_name in queued_imgs:
        typer.echo(f"Processing {mask_name}...")

        # Process for all zones
        zone_results = process_mask_pair(
            mask_name,
            arteries_dir,
            veins_dir,
            discs_dir,
            diagnosis_data,
            zones_to_process,
        )

        # Remove file extension from mask_name
        mask_base_name = os.path.splitext(mask_name)[0]

        # Organize results by zone
        for zone in zones_to_process:
            zone_data = zone_results[zone]
            results_by_zone[zone].append(
                {
                    "mask_name": mask_base_name,
                    "diagnosis": zone_data["diagnosis"],
                    "MAD": zone_data["MAD"],
                    "MVD": zone_data["MVD"],
                    "AVR": zone_data["AVR"],
                    "LAD": zone_data["LAD"],
                    "LVD": zone_data["LVD"],
                    "LAVR": zone_data["LAVR"],
                }
            )

            # Print results for this zone
            zone_display = (
                "Zone " + zone.upper() if zone != "disc" else zone.capitalize()
            )
            typer.echo(
                f"  {zone_display} - MAD: {zone_data['MAD']:.2f}, MVD: {zone_data['MVD']:.2f}, "
                f"AVR: {zone_data['AVR']:.3f}, LAD: {zone_data['LAD']:.2f}, "
                f"LVD: {zone_data['LVD']:.2f}, LAVR: {zone_data['LAVR']:.3f}"
            )

    typer.echo(f"\nSuccessfully processed {len(queued_imgs)} mask pairs for all zones")

    # Create DataFrames and save results for each zone
    output_dir.mkdir(exist_ok=True)

    for zone in zones_to_process:
        df_zone = pd.DataFrame(results_by_zone[zone])

        # Create filename based on zone
        if zone == "disc":
            filename = "avr_results_disc.csv"
        else:
            filename = f"avr_results_zone_{zone}.csv"

        output_path = output_dir / filename
        df_zone.to_csv(output_path, index=False)

        # Display summary statistics for this zone
        zone_display = "Zone " + zone.upper() if zone != "disc" else zone.capitalize()
        typer.echo(f"\n=== {zone_display.upper()} RESTRICTED RESULTS ===")
        typer.echo(
            df_zone[["MAD", "MVD", "AVR", "LAD", "LVD", "LAVR"]].describe().to_string()
        )

        # Display first few rows
        typer.echo(f"\n=== {zone_display.upper()} DETAILED RESULTS (First 10) ===")
        typer.echo(df_zone.head(10).to_string())

        typer.echo(f"\n{zone_display} results saved to: {output_path}")

    typer.echo(f"\nTotal samples processed: {len(queued_imgs)}")
    typer.echo(f"CSV files saved in: {output_dir}")


if __name__ == "__main__":
    app()
