# -*- coding: utf-8 -*-
"""
Convert 2D PNG stacks (2D_data/<ds>/<split>/{CT,Mask}/<patient>/*.png)
into per-patient NPZ volumes consumable by training.dataset.vos_raw_dataset.NPZRawDataset.

Output NPZ format (matches FLARE25 RECIST training npz):
    imgs  : (D, H, W) uint8, values in [0, 255]        -- grayscale CT
    gts   : (D, H, W) uint8, values in {0, 1}           -- binary tumor mask
    middle_idx : () int32                               -- index of the middle slice (bookkeeping only,
                                                          unused by upstream training pipeline)
    patient_name : () str                               -- for traceability
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _read_png(path: str) -> np.ndarray:
    """Read a PNG, normalize RGBA/RGB -> single grayscale channel (uint8)."""
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # For this project's data the R/G/B channels are identical (grayscale stored as RGBA).
        # Drop alpha if present and take the first channel.
        return arr[..., 0]
    raise ValueError(f"Unexpected image ndim={arr.ndim} for {path}")


def _sort_slices(filenames):
    """Sort slice filenames by their numeric stem when possible, else lexicographic."""
    def key(fn):
        stem = os.path.splitext(fn)[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    return sorted(filenames, key=key)


def convert_patient(ct_dir: str, mask_dir: str, patient: str) -> dict | None:
    """Load all slices of one patient into imgs/gts arrays; return None if mismatched or empty."""
    ct_folder = os.path.join(ct_dir, patient)
    msk_folder = os.path.join(mask_dir, patient)

    ct_files = [f for f in os.listdir(ct_folder) if f.lower().endswith(".png")]
    ct_files = _sort_slices(ct_files)

    kept_names, imgs, gts = [], [], []
    for fn in ct_files:
        ct_path = os.path.join(ct_folder, fn)
        msk_path = os.path.join(msk_folder, fn)
        if not os.path.exists(msk_path):
            continue  # paired mask missing → skip this slice
        img = _read_png(ct_path)
        msk = _read_png(msk_path)
        if img.shape != msk.shape:
            print(f"  [WARN] shape mismatch for {patient}/{fn}: img={img.shape} msk={msk.shape}; skipped")
            continue
        imgs.append(img.astype(np.uint8))
        gts.append((msk > 127).astype(np.uint8))
        kept_names.append(fn)

    if len(imgs) == 0:
        return None

    imgs_arr = np.stack(imgs, axis=0)  # (D, H, W) uint8
    gts_arr = np.stack(gts, axis=0)  # (D, H, W) uint8
    # Middle slice index (match the test_medsam2_2d_video convention: filename sort order).
    middle_idx = len(kept_names) // 2
    return {
        "imgs": imgs_arr,
        "gts": gts_arr,
        "middle_idx": np.int32(middle_idx),
        "patient_name": patient,
        "slice_filenames": np.array(kept_names),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["colon", "kits", "lits", "pancreas", "local"])
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--src", default="./2D_data")
    ap.add_argument(
        "--dst",
        default="./MedSAM2/data/tumorseg_npz",
        help="Output root; actual path = <dst>/<dataset>/<split>/<patient>.npz",
    )
    ap.add_argument("--min-slices", type=int, default=4, help="Skip patients with fewer slices")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ct_dir = os.path.join(args.src, args.dataset, args.split, "CT")
    mask_dir = os.path.join(args.src, args.dataset, args.split, "Mask")
    if not os.path.isdir(ct_dir):
        print(f"[ERR] missing {ct_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(args.dst, args.dataset, args.split)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    patients = sorted(p for p in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, p)))
    kept = skipped_small = skipped_missing = 0
    d_distribution = []

    for pat in patients:
        out_path = os.path.join(out_dir, f"{pat}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            kept += 1
            data = np.load(out_path)
            d_distribution.append(int(data["imgs"].shape[0]))
            continue

        data = convert_patient(ct_dir, mask_dir, pat)
        if data is None:
            print(f"  [SKIP] {pat}: no paired slices")
            skipped_missing += 1
            continue
        D = data["imgs"].shape[0]
        if D < args.min_slices:
            print(f"  [SKIP] {pat}: only {D} slices < min_slices={args.min_slices}")
            skipped_small += 1
            continue

        np.savez_compressed(out_path, **data)
        kept += 1
        d_distribution.append(D)

    d_arr = np.array(d_distribution) if d_distribution else np.array([0])
    print(
        f"\n[{args.dataset}/{args.split}] wrote {kept} patients | "
        f"skipped {skipped_small} for <{args.min_slices} slices, {skipped_missing} for missing pairs\n"
        f"  D: min={d_arr.min()} max={d_arr.max()} mean={d_arr.mean():.1f} median={int(np.median(d_arr))}\n"
        f"  out_dir: {out_dir}"
    )


if __name__ == "__main__":
    main()
