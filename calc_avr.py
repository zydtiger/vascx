"""
Calculate Arteriole-Venular Ratios from retinal vessel masks.

This script loads arteriole and venular mask images, calculates both mean and largest diameters
restricted to Zone A (1.0-1.5 disc diameters from optic disc center), and
computes both mean and largest arteriole-venular ratios.
Outputs MAD (Mean Artery Diameter), MVD (Mean Vein Diameter), AVR (Mean Ratio),
LAD (Largest Artery Diameter), LVD (Largest Vein Diameter), and LAVR (Largest Ratio).
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from skimage import io, measure
from skimage.morphology import skeletonize
from scipy import ndimage
import typer
from typing import Optional

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


def create_zone_a_mask(
    disc_mask: np.ndarray,
    inner_radius_factor: float = 1.0,
    outer_radius_factor: float = 1.5,
) -> np.ndarray:
    """Create a binary mask for Zone A (annular region between 1.0-1.5 disc diameters)."""
    # Find disc center and radius
    labeled_disc = measure.label(disc_mask)
    regions = measure.regionprops(labeled_disc)

    # Get the largest region (should be the disc)
    largest_region = max(regions, key=lambda r: r.area)
    center_y, center_x = largest_region.centroid

    # Calculate disc radius as average of minor and major axes lengths divided by 2
    disc_radius = (
        largest_region.major_axis_length + largest_region.minor_axis_length
    ) / 4

    # Create coordinate grids
    height, width = disc_mask.shape
    y, x = np.ogrid[:height, :width]

    # Calculate distance from center for each pixel
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create Zone A mask (annular region between inner_radius_factor and outer_radius_factor * disc_radius)
    inner_radius = inner_radius_factor * disc_radius
    outer_radius = outer_radius_factor * disc_radius

    zone_a_mask = (distances >= inner_radius) & (distances <= outer_radius)

    return zone_a_mask


def calculate_mean_diameter(
    mask: np.ndarray,
    disc_mask: Optional[np.ndarray] = None,
    restrict_to_zone_a: bool = True,
) -> float:
    """Calculate mean vessel diameter from a binary mask using distance transform and skeleton.

    Args:
        mask: Binary mask of vessels
        disc_mask: Binary mask of optic disc (required if restrict_to_zone_a=True)
        restrict_to_zone_a: If True, only consider vessels in Zone A (1.0-1.5 disc diameters)
    """
    # Create Zone A mask if restriction is enabled
    if restrict_to_zone_a and disc_mask is not None:
        zone_a_mask = create_zone_a_mask(disc_mask)
        # Apply Zone A restriction to vessel mask
        restricted_mask = mask & zone_a_mask
    else:
        restricted_mask = mask

    # Calculate distance transform
    distance = ndimage.distance_transform_edt(restricted_mask)

    # Extract skeleton (centerline) of vessels
    skeleton = skeletonize(restricted_mask > 0)
    # Get distances only from centerline pixels
    vessel_distances = distance[skeleton]

    # Diameter is 2 * radius (distance transform gives radius)
    diameters = vessel_distances * 2

    # Return mean diameter
    return np.mean(diameters)


def calculate_largest_diameter(
    mask: np.ndarray,
    disc_mask: Optional[np.ndarray] = None,
    restrict_to_zone_a: bool = True,
) -> float:
    """Calculate largest vessel diameter from a binary mask using distance transform and skeleton.

    Args:
        mask: Binary mask of vessels
        disc_mask: Binary mask of optic disc (required if restrict_to_zone_a=True)
        restrict_to_zone_a: If True, only consider vessels in Zone A (1.0-1.5 disc diameters)
    """
    # Create Zone A mask if restriction is enabled
    if restrict_to_zone_a and disc_mask is not None:
        zone_a_mask = create_zone_a_mask(disc_mask)
        # Apply Zone A restriction to vessel mask
        restricted_mask = mask & zone_a_mask
    else:
        restricted_mask = mask

    # Calculate distance transform
    distance = ndimage.distance_transform_edt(restricted_mask)

    # Extract skeleton (centerline) of vessels
    skeleton = skeletonize(restricted_mask > 0)
    # Get distances only from centerline pixels
    vessel_distances = distance[skeleton]

    # Diameter is 2 * radius (distance transform gives radius)
    diameters = vessel_distances * 2

    # Return largest diameter
    return np.max(diameters)


def find_matching_files(arteries_dir: Path, veins_dir: Path) -> list:
    """Find matching mask files in arteries and veins directories."""
    artery_files = [
        f
        for f in os.listdir(arteries_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
    ]
    vein_files = [
        f
        for f in os.listdir(veins_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
    ]

    # Find common files
    common_files = list(set(artery_files) & set(vein_files))

    return sorted(common_files)


def get_diagnosis(
    mask_name: str, diagnosis_data: Optional[pd.DataFrame]
) -> Optional[str]:
    """Get diagnosis for a given mask name from the diagnosis dataframe."""
    if diagnosis_data is None:
        return None

    # Remove file extension from mask_name
    mask_base_name = os.path.splitext(mask_name)[0]
    matching_row = diagnosis_data[
        diagnosis_data["Identifier"].astype(str) == mask_base_name
    ]
    return matching_row["Diagnosis5"].iloc[0]


def process_mask_pair(
    mask_name: str,
    arteries_dir: Path,
    veins_dir: Path,
    discs_dir: Path,
    diagnosis_data: Optional[pd.DataFrame],
    restrict_to_zone_a: bool = True,
) -> tuple:
    """Process a pair of masks and return MAD, MVD, AVR, LAD, LVD, LAVR, and diagnosis."""
    # Load masks
    artery_mask = load_mask_image(arteries_dir / mask_name)
    vein_mask = load_mask_image(veins_dir / mask_name)
    disc_mask = load_disc_mask(mask_name, discs_dir) if restrict_to_zone_a else None

    # Calculate mean diameters using skeleton and Zone A restriction
    MAD = calculate_mean_diameter(artery_mask, disc_mask, restrict_to_zone_a)
    MVD = calculate_mean_diameter(vein_mask, disc_mask, restrict_to_zone_a)

    # Calculate AVR (Arteriole-Venular Ratio)
    AVR = MAD / MVD

    # Calculate largest diameters using skeleton and Zone A restriction
    LAD = calculate_largest_diameter(artery_mask, disc_mask, restrict_to_zone_a)
    LVD = calculate_largest_diameter(vein_mask, disc_mask, restrict_to_zone_a)

    # Calculate LAVR (Largest Arteriole-Venular Ratio)
    LAVR = LAD / LVD

    # Get diagnosis
    diagnosis = get_diagnosis(mask_name, diagnosis_data)

    return MAD, MVD, AVR, LAD, LVD, LAVR, diagnosis


@app.command()
def main(
    arteries_dir: Path = typer.Option(
        Path("./arteries"),
        "--arteries",
        help="Directory containing arteriole mask images",
    ),
    veins_dir: Path = typer.Option(
        Path("./veins"), "--veins", help="Directory containing venular mask images"
    ),
    discs_dir: Path = typer.Option(
        Path("./discs"), "--discs", help="Directory containing optic disc mask images"
    ),
    output_csv: Path = typer.Option(
        Path("avr_lavr_results.csv"),
        "--output",
        help="Output CSV file for AVR and LAVR results",
    ),
):
    """Calculate AVR and LAVR from retinal vessel masks restricted to Zone A."""

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

    # Try to load diagnosis data if available
    diagnosis_data = None
    data_file = Path("Data.xlsx")
    if data_file.exists():
        try:
            diagnosis_data = pd.read_excel(data_file)
            typer.echo(f"Loaded diagnosis data from {data_file}")
        except Exception as e:
            typer.echo(f"Warning: Could not load {data_file}: {e}")

    # Find matching mask files
    matching_files = find_matching_files(arteries_dir, veins_dir)
    typer.echo(f"Found {len(matching_files)} matching mask files")

    if len(matching_files) == 0:
        typer.echo(
            "No matching mask files found. Please check the directories.", err=True
        )
        raise typer.Exit(1)

    typer.echo("Matching files:")
    for file in matching_files[:5]:  # Show first 5 files
        typer.echo(f"  - {file}")
    if len(matching_files) > 5:
        typer.echo(f"  ... and {len(matching_files) - 5} more files")

    # Initialize results list
    results_zone_a = []

    # Process each matching mask pair using Zone A restriction
    for mask_name in matching_files:
        typer.echo(f"Processing {mask_name}...")

        # Process with Zone A restriction (skeleton-only method)
        MAD_zone, MVD_zone, AVR_zone, LAD_zone, LVD_zone, LAVR_zone, diagnosis = (
            process_mask_pair(
                mask_name,
                arteries_dir,
                veins_dir,
                discs_dir,
                diagnosis_data,
                restrict_to_zone_a=True,
            )
        )

        # Remove file extension from mask_name
        mask_base_name = os.path.splitext(mask_name)[0]

        results_zone_a.append(
            {
                "mask_name": int(mask_base_name),
                "diagnosis": diagnosis,
                "MAD": MAD_zone,
                "MVD": MVD_zone,
                "AVR": AVR_zone,
                "LAD": LAD_zone,
                "LVD": LVD_zone,
                "LAVR": LAVR_zone,
            }
        )

        typer.echo(
            f"  Zone A - MAD: {MAD_zone:.2f}, MVD: {MVD_zone:.2f}, AVR: {AVR_zone:.3f}, LAD: {LAD_zone:.2f}, LVD: {LVD_zone:.2f}, LAVR: {LAVR_zone:.3f}"
        )

    typer.echo(
        f"\nSuccessfully processed {len(results_zone_a)} mask pairs with Zone A restriction"
    )

    # Create DataFrame and save results
    df_zone_a = pd.DataFrame(results_zone_a)

    # Sort by mask_name
    df_zone_a = df_zone_a.sort_values("mask_name").reset_index(drop=True)

    # Display summary statistics
    typer.echo("\n=== ZONE A RESTRICTED RESULTS ===")
    typer.echo(
        df_zone_a[["MAD", "MVD", "AVR", "LAD", "LVD", "LAVR"]].describe().to_string()
    )

    # Display first few rows
    typer.echo("\n=== DETAILED RESULTS (First 10) ===")
    typer.echo(df_zone_a.head(10).to_string())

    # Save results as CSV
    df_zone_a.to_csv(output_csv, index=False)

    typer.echo(f"\nResults saved to: {output_csv}")
    typer.echo(f"Total samples processed: {len(df_zone_a)}")


if __name__ == "__main__":
    app()
