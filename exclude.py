import os
from PIL import Image
import numpy as np


artery_path = "./data/ODIR/arteries"
vein_path = "./data/ODIR/veins"
disc_path = "./data/ODIR/discs"

overlap_path = "./data/ODIR/artery_vein_overlap"
os.makedirs(overlap_path, exist_ok=True)


def load_mask(path):
    """
    Load as grayscale, return numpy array uint8 (H,W).
    If file missing, return None.
    """
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


for fname in os.listdir(artery_path):
    if not fname.lower().endswith(".png"):
        continue

    artery_file = os.path.join(artery_path, fname)
    vein_file = os.path.join(vein_path, fname)
    disc_file = os.path.join(disc_path, fname)

    # load all 3
    artery_mask = load_mask(artery_file)
    vein_mask = load_mask(vein_file)
    disc_mask = load_mask(disc_file)

    # if any counterpart is missing, skip
    if artery_mask is None or vein_mask is None or disc_mask is None:
        print(f"[skip: missing match] {fname}")
        continue

    # check disc is all black
    if np.max(disc_mask) == 0:
        print(f"[skip: no disc zone] {fname}")
        continue

    # otherwise generate overlap
    # Strategy:
    #   - normalize artery_mask to {0,1}
    #   - normalize vein_mask to {0,1}
    #   - overlap = (artery OR vein) * 255
    # If you want additive (different intensities for art/vein),
    # you can adjust below.

    art_bin = (artery_mask > 0).astype(np.uint8)
    vein_bin = (vein_mask > 0).astype(np.uint8)

    overlap_bin = np.clip(art_bin | vein_bin, 0, 1)
    if overlap_bin.sum() < 65000:
        continue
    overlap_img = Image.fromarray((overlap_bin * 255).astype(np.uint8), mode="L")

    save_file = os.path.join(overlap_path, fname)
    overlap_img.save(save_file)

    print(f"[saved] {save_file}")
