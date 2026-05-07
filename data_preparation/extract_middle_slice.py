#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traverse datasets under a base root like:
  <ROOT>/<DATASET>/<SPLIT>/{CT,Mask}/<PATIENT>/*.png
For each existing split (train/test[/val/validation]), write:
  <ROOT>/<DATASET>/<SPLIT>/annotation_dict_middle.json

JSON format:
{
  "<patient_name>": {
    "middle_filename": "056.png",
    "ct_path":   "/abs/.../<DATASET>/<SPLIT>/CT/<patient>/056.png",
    "mask_path": "/abs/.../<DATASET>/<SPLIT>/Mask/<patient>/056.png"  # may be null if no mask
  },
  ...
}

Selection rule:
- Prefer the exact middle (by numeric-aware sort of filenames).
- If the middle has no mask but others do, pick the nearest file that has mask.
- If no mask exists for the patient, pick the CT middle and set mask_path=None.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

VALID_EXTS = (".png")
DEFAULT_SPLITS = ["train", "test"]

def numeric_key(name: str) -> Tuple[int, str]:
    """Sort by last number in filename if present, fallback to lexicographic."""
    nums = re.findall(r'\d+', name)
    last = int(nums[-1]) if nums else -1
    return (last, name)

def list_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        [f for f in os.listdir(folder)
         if f.lower().endswith(VALID_EXTS) and os.path.isfile(os.path.join(folder, f))],
        key=numeric_key
    )

def pick_middle_with_mask(ct_files: List[str], ct_dir: str, mask_dir: str) -> Tuple[str, Optional[str]]:
    """
    Return (middle_ct_path, middle_mask_path or None).
    If exact middle has no mask, search nearest neighbors with mask.
    """
    if not ct_files:
        return "", None
    mid_idx = len(ct_files) // 2
    # search order: exact middle, then expand left/right by distance
    candidates = list(range(len(ct_files)))
    candidates.sort(key=lambda i: abs(i - mid_idx))

    chosen_ct = None
    chosen_mask = None

    for i in candidates:
        fname = ct_files[i]
        ct_path = os.path.join(ct_dir, fname)
        mask_path = os.path.join(mask_dir, fname) if os.path.isdir(mask_dir) else None
        if mask_path and os.path.exists(mask_path):
            chosen_ct = ct_path
            chosen_mask = mask_path
            break

    # if no mask at all, fall back to CT middle
    if chosen_ct is None:
        fname_mid = ct_files[mid_idx]
        chosen_ct = os.path.join(ct_dir, fname_mid)
        chosen_mask = None

    return chosen_ct, chosen_mask

def detect_splits(dataset_dir: str) -> List[str]:
    """
    Return a list of splits that actually exist and contain a CT folder.
    Prioritize DEFAULT_SPLITS order; also include any other subdirs that have CT.
    """
    splits = []
    entries = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    # add standard splits if valid
    for sp in DEFAULT_SPLITS:
        ct_dir = os.path.join(dataset_dir, sp, "CT")
        if os.path.isdir(ct_dir):
            splits.append(sp)
    # add any other subdir that has CT but not already included
    for d in entries:
        if d in splits:
            continue
        ct_dir = os.path.join(dataset_dir, d, "CT")
        if os.path.isdir(ct_dir):
            splits.append(d)
    return splits

def process_split(root: str, dataset: str, split: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Build mapping for one split:
    { patient_name: { middle_filename, ct_path, mask_path(or None) } }
    """
    split_dir = os.path.join(root, dataset, split)
    ct_root   = os.path.join(split_dir, "CT")
    mask_root = os.path.join(split_dir, "Mask")

    if not os.path.isdir(ct_root):
        return {}

    mapping: Dict[str, Dict[str, Optional[str]]] = {}

    for patient in sorted(os.listdir(ct_root)):
        patient_ct_dir   = os.path.join(ct_root, patient)
        patient_mask_dir = os.path.join(mask_root, patient)

        if not os.path.isdir(patient_ct_dir):
            continue

        ct_files = list_image_files(patient_ct_dir)
        if not ct_files:
            continue

        ct_path, mask_path = pick_middle_with_mask(ct_files, patient_ct_dir, patient_mask_dir)
        if not ct_path:
            continue

        middle_filename = os.path.basename(ct_path)
        mapping[patient] = {
            "middle_filename": middle_filename,
            "ct_path":   os.path.abspath(ct_path),
            "mask_path": os.path.abspath(mask_path) if mask_path else None,
        }

    return mapping

def write_json(path: str, data: Dict) -> None:
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def auto_detect_datasets(root: str) -> List[str]:
    """
    Return dataset directory names directly under <root> (e.g., 'lits', ...),
    excluding non-dirs and hidden dirs.
    """
    if not os.path.isdir(root):
        return []
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    ])

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build middle-slice JSONs for all datasets/splits.")
    ap.add_argument("--root", type=str, default="./2D_data",
                    help="Base root, e.g., ./2D_data")
    ap.add_argument("--datasets", type=str, default="",
                    help="Comma-separated dataset names (optional). If empty, auto-detect all under --root.")
    ap.add_argument("--outfile", type=str, default="annotation_dict_middle.json",
                    help="Output JSON filename to place under each <dataset>/<split>/")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] \
               if args.datasets else auto_detect_datasets(root)

    if not datasets:
        print(f"[WARN] No datasets found under: {root}")
        return

    print(f"[INFO] Datasets to process: {datasets}")

    total_written = 0
    for ds in datasets:
        dataset_dir = os.path.join(root, ds)
        if not os.path.isdir(dataset_dir):
            print(f"[WARN] Skip non-dir dataset: {dataset_dir}")
            continue

        splits = detect_splits(dataset_dir)
        if not splits:
            print(f"[WARN] No valid splits found in {dataset_dir}")
            continue

        for sp in splits:
            mapping = process_split(root, ds, sp)
            out_json = os.path.join(root, ds, sp, args.outfile)
            write_json(out_json, mapping)
            print(f"[OK] {ds}/{sp}: wrote {len(mapping)} patients -> {out_json}")
            total_written += 1

    print(f"[DONE] Wrote {total_written} JSON files.")

if __name__ == "__main__":
    main()
